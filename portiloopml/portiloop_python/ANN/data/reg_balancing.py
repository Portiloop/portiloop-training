
import torch
import numpy as np
import pandas as pd
from random import randint, seed
import logging
import copy
from scipy.ndimage import gaussian_filter1d, convolve1d



def get_lds_kernel(ks, sigma):
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    return kernel_window


def generate_label_distribution_and_lds(dataset, kernel_size=5, kernel_std=2.0, nb_bins=100, reweight='inv_sqrt'):
    """
    Returns:
        distribution: the distribution of labels in the dataset
        lds: the same distribution, smoothed with a gaussian kernel
    """

    weights = torch.tensor([0.3252, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0069, 0.0163,
                            0.0000, 0.0366, 0.0000, 0.0179, 0.0000, 0.0076, 0.0444, 0.0176, 0.0025,
                            0.0056, 0.0000, 0.0416, 0.0039, 0.0000, 0.0000, 0.0000, 0.0171, 0.0000,
                            0.0000, 0.0042, 0.0114, 0.0209, 0.0023, 0.0036, 0.0106, 0.0241, 0.0034,
                            0.0000, 0.0056, 0.0000, 0.0029, 0.0241, 0.0076, 0.0027, 0.0012, 0.0000,
                            0.0166, 0.0028, 0.0000, 0.0000, 0.0000, 0.0197, 0.0000, 0.0000, 0.0021,
                            0.0054, 0.0191, 0.0014, 0.0023, 0.0074, 0.0000, 0.0186, 0.0000, 0.0088,
                            0.0000, 0.0032, 0.0135, 0.0069, 0.0029, 0.0016, 0.0164, 0.0068, 0.0022,
                            0.0000, 0.0000, 0.0000, 0.0191, 0.0000, 0.0000, 0.0017, 0.0082, 0.0181,
                            0.0019, 0.0038, 0.0064, 0.0000, 0.0133, 0.0000, 0.0069, 0.0000, 0.0025,
                            0.0186, 0.0076, 0.0031, 0.0016, 0.0218, 0.0105, 0.0049, 0.0000, 0.0000,
                            0.0246], dtype=torch.float64)

    lds = None
    dist = None
    bins = None
    return weights, dist, lds, bins

    # TODO: remove before

    dataset_len = len(dataset)
    logging.debug(f"Length of the dataset passed to generate_label_distribution_and_lds: {dataset_len}")
    logging.debug(f"kernel_size: {kernel_size}")
    logging.debug(f"kernel_std: {kernel_std}")
    logging.debug(f"Generating empirical distribution...")

    tab = np.array([dataset[i][3].item() for i in range(dataset_len)])
    tab = np.around(tab, decimals=5)
    elts = np.unique(tab)
    logging.debug(f"all labels: {elts}")
    dist, bins = np.histogram(tab, bins=nb_bins, density=False, range=(0.0, 1.0))

    # dist, bins = np.histogram([dataset[i][3].item() for i in range(dataset_len)], bins=nb_bins, density=False, range=(0.0, 1.0))

    logging.debug(f"dist: {dist}")

    # kernel = get_lds_kernel(kernel_size, kernel_std)
    # lds = convolve1d(dist, weights=kernel, mode='constant')

    lds = gaussian_filter1d(input=dist, sigma=kernel_std, axis=- 1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

    weights = np.sqrt(lds) if reweight == 'inv_sqrt' else lds
    # scaling = len(weights) / np.sum(weights)  # not the same implementation as in the original repo
    scaling = 1.0 / np.sum(weights)
    weights = weights * scaling

    return weights, dist, lds, bins


class LabelDistributionSmoothing:
    def __init__(self, c=1.0, dataset=None, weights=None, kernel_size=5, kernel_std=2.0, nb_bins=100, weighting_mode="inv_sqrt"):
        """
        If provided, lds_distribution must be a numpy.array representing a density over [0.0, 1.0] (e.g. first element of a numpy.histogram)
        When lds_distribution is provided, it overrides everything else
        c is the scaling constant for lds weights
        weighting_mode can be 'inv' or 'inv_sqrt'
        """
        assert dataset is not None or weights is not None, "Either a dataset or weights must be provided"
        self.distribution = None
        self.bins = None
        self.lds_distribution = None
        if weights is None:
            self.weights, self.distribution, self.lds_distribution, self.bins = generate_label_distribution_and_lds(dataset, kernel_size, kernel_std,
                                                                                                                    nb_bins, weighting_mode)
            logging.debug(f"self.distribution: {self.weights}")
            logging.debug(f"self.lds_distribution: {self.weights}")
        else:
            self.weights = weights
        self.nb_bins = len(self.weights)
        self.bin_width = 1.0 / self.nb_bins
        self.c = c
        logging.debug(f"The LDS distribution has {self.nb_bins} bins of width {self.bin_width}")
        self.weights = torch.tensor(self.weights)

        logging.debug(f"self.weights: {self.weights}")

    def lds_weights_batch(self, batch_labels):
        device = batch_labels.device
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        last_bin = 1.0 - self.bin_width
        batch_idxs = torch.minimum(batch_labels, torch.ones_like(batch_labels) * last_bin) / self.bin_width  # FIXME : double check
        batch_idxs = batch_idxs.floor().long()
        res = 1.0 / self.weights[batch_idxs]
        return res

    def __str__(self):
        return f"LDS nb_bins: {self.nb_bins}\nbins: {self.bins}\ndistribution: {self.distribution}\nlds_distribution: {self.lds_distribution}\nweights: {self.weights} "


class SurpriseReweighting:
    """
    Custom reweighting Yann
    """

    def __init__(self, weights=None, nb_bins=100, alpha=1e-3):
        if weights is None:
            self.weights = [1.0, ] * nb_bins
            self.weights = torch.tensor(self.weights)
            self.weights = self.weights / torch.sum(self.weights)
        else:
            self.weights = weights
        self.weights = self.weights.detach()
        self.nb_bins = len(self.weights)
        self.bin_width = 1.0 / self.nb_bins
        self.alpha = alpha
        logging.debug(f"The SR distribution has {self.nb_bins} bins of width {self.bin_width}")
        logging.debug(f"Initial self.weights: {self.weights}")

    def update_and_get_weighted_loss(self, batch_labels, unweighted_loss):
        device = batch_labels.device
        if self.weights.device != device:
            logging.debug(f"Moving SR weights to {device}")
            self.weights = self.weights.to(device)
        last_bin = 1.0 - self.bin_width
        batch_idxs = torch.minimum(batch_labels, torch.ones_like(batch_labels) * last_bin) / self.bin_width  # FIXME : double check
        batch_idxs = batch_idxs.floor().long()
        self.weights = self.weights.detach()  # ensure no gradients
        weights = copy.deepcopy(self.weights[batch_idxs])
        res = unweighted_loss * weights
        with torch.no_grad():
            abs_loss = torch.abs(unweighted_loss)

            # compute the mean loss per idx

            num = torch.zeros(self.nb_bins, device=device)
            num = num.index_add(0, batch_idxs, abs_loss)
            bincount = torch.bincount(batch_idxs, minlength=self.nb_bins)
            div = bincount.float()
            idx_unchanged = bincount == 0
            idx_changed = bincount != 0
            div[idx_unchanged] = 1.0
            mean_loss_per_idx_normalized = num / div
            sum_changed_weights = torch.sum(self.weights[idx_changed])
            sum_mean_loss = torch.sum(mean_loss_per_idx_normalized[idx_changed])
            mean_loss_per_idx_normalized[idx_changed] = mean_loss_per_idx_normalized[idx_changed] * sum_changed_weights / sum_mean_loss
            # logging.debug(f"old self.weights: {self.weights}")
            self.weights[idx_changed] = (1.0 - self.alpha) * self.weights[idx_changed] + self.alpha * mean_loss_per_idx_normalized[idx_changed]
            self.weights /= torch.sum(self.weights)  # force sum to 1
            # logging.debug(f"unique_idx: {unique_idx}")
            # logging.debug(f"new self.weights: {self.weights}")
            # logging.debug(f"new torch.sum(self.weights): {torch.sum(self.weights)}")
        return torch.sqrt(res * self.nb_bins)

    def __str__(self):
        return f"LDS nb_bins: {self.nb_bins}\nweights: {self.weights}"