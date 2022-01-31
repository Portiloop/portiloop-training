from pathlib import Path
from random import randint, seed
import numpy as np
import pandas as pd
import logging

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch
from sklearn.model_selection import train_test_split


# Global Variable for names of file
filename_regression_dataset = f"dataset_regression_{PHASE}_big_250_matlab_standardized_envelope_pf.txt"
filename_classification_dataset = f"dataset_classification_{PHASE}_big_250_matlab_standardized_envelope_pf.txt"
subject_list = f"subject_sequence_{PHASE}_big.txt"
subject_list_p1 = f"subject_sequence_p1_big.txt"
subject_list_p2 = f"subject_sequence_p2_big.txt"


class SignalDataset(Dataset):
    def __init__(self, filename, path, window_size, fe, seq_len, seq_stride, list_subject, len_segment):
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename

        self.data = pd.read_csv(self.path_file, header=None).to_numpy()
        assert list_subject is not None
        used_sequence = np.hstack(
            [range(int(s[1]), int(s[2])) for s in list_subject])
        # 115+30 = nb seconds per sequence in the dataset
        split_data = np.array(np.split(self.data, int(
            len(self.data) / (len_segment + 30 * fe))))
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape(
            (split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(
            self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_envelope = torch.tensor(self.data[1], dtype=torch.float)
        self.seq_len = seq_len  # 1 means single sample / no sequence ?
        self.idx_stride = seq_stride
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here
        total_spindles = np.sum(self.data[3] > THRESHOLD)
        logging.debug(
            f"total number of spindles in this dataset : {total_spindles}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size -
                            1] >= 0, f"Bad index: {idx}."

        signal_seq = self.full_signal[idx - (self.past_signal_len - self.idx_stride)
                                             :idx + self.window_size].unfold(0, self.window_size, self.idx_stride)
        envelope_seq = self.full_envelope[idx - (self.past_signal_len - self.idx_stride)
                                                 :idx + self.window_size].unfold(0, self.window_size, self.idx_stride)

        ratio_pf = torch.tensor(
            self.data[2][idx + self.window_size - 1], dtype=torch.float)
        label = torch.tensor(
            self.data[3][idx + self.window_size - 1], dtype=torch.float)

        return signal_seq, envelope_seq, ratio_pf, label

    def is_spindle(self, idx):
        assert 0 <= idx <= len(
            self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > THRESHOLD) else False


class UnlabelledSignalDatasetSingleSegment(Dataset):
    """
    Caution: this dataset does not sample sequences, but single windows
    """

    def __init__(self, unlabelled_segment, window_size):
        self.window_size = window_size
        self.full_signal = torch.tensor(
            unlabelled_segment, dtype=torch.float).squeeze()
        assert len(
            self.full_signal.shape) == 1, f"Segment has more than one dimension: {self.full_signal.shape}"
        assert self.window_size <= len(
            self.full_signal), "Segment smaller than window size."
        self.seq_len = 1  # 1 means single sample / no sequence ?
        self.idx_stride = 1
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.full_signal) - self.window_size)  # all possible idxs in the dataset
                        if (not idx < self.past_signal_len)]  # far enough from the beginning to build a sequence up to here

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        signal_seq = self.full_signal[idx:idx +
                                      self.window_size].unfold(0, self.window_size, 1)
        true_idx = idx + self.window_size
        return signal_seq, true_idx


def get_class_idxs(dataset, distribution_mode):
    """
    Directly outputs idx_true and idx_false arrays
    """
    length_dataset = len(dataset)

    nb_true = 0
    nb_false = 0

    idx_true = []
    idx_false = []

    for i in range(length_dataset):
        is_spindle = dataset.is_spindle(i)
        if is_spindle or distribution_mode == 1:
            nb_true += 1
            idx_true.append(i)
        else:
            nb_false += 1
            idx_false.append(i)

    assert len(dataset) == nb_true + nb_false, f"Bad length dataset"

    return np.array(idx_true), np.array(idx_false)


# Sampler avec liste et sans rand liste

class RandomSampler(Sampler):
    """
    Samples elements randomly and evenly between the two classes.
    The sampling happens WITH replacement.
    __iter__ stops after an arbitrary number of iterations = batch_size_list * nb_batch
    Arguments:
      idx_true: np.array
      idx_false: np.array
      batch_size (int)
      nb_batch (int, optional): number of iteration before end of __iter__(), this defaults to len(data_source)
    """

    def __init__(self, idx_true, idx_false, batch_size, distribution_mode, nb_batch):
        self.idx_true = idx_true
        self.idx_false = idx_false
        self.nb_true = self.idx_true.size
        self.nb_false = self.idx_false.size
        self.length = nb_batch * batch_size
        self.distribution_mode = distribution_mode

    def __iter__(self):
        global precision_validation_factor
        global recall_validation_factor
        cur_iter = 0
        seed()
        # epsilon = 1e-7 proba = float(0.5 + 0.5 * (precision_validation_factor - recall_validation_factor) / (precision_validation_factor +
        # recall_validation_factor + epsilon))
        proba = 0.5
        if self.distribution_mode == 1:
            proba = 1
        logging.debug(f"proba: {proba}")

        while cur_iter < self.length:
            cur_iter += 1
            sample_class = np.random.choice([0, 1], p=[1 - proba, proba])
            if sample_class:  # sample true
                idx_file = randint(0, self.nb_true - 1)
                idx_res = self.idx_true[idx_file]
            else:  # sample false
                idx_file = randint(0, self.nb_false - 1)
                idx_res = self.idx_false[idx_file]

            yield idx_res

    def __len__(self):
        return self.length


# Sampler validation

class ValidationSampler(Sampler):
    """
    network_stride (int >= 1, default: 1): divides the size of the dataset (and of the batch) by striding further than 1
    """

    def __init__(self, data_source, seq_stride, nb_segment, len_segment, network_stride):
        network_stride = int(network_stride)
        assert network_stride >= 1
        self.network_stride = network_stride
        self.seq_stride = seq_stride
        self.data = data_source
        self.nb_segment = nb_segment
        self.len_segment = len_segment

    def __iter__(self):
        seed()
        # len sequence = 115 s + add the 15 first s?
        batches_per_segment = self.len_segment // self.seq_stride
        cursor_batch = 0
        while cursor_batch < batches_per_segment:
            for i in range(self.nb_segment):
                for j in range(0, (self.seq_stride // self.network_stride) * self.network_stride, self.network_stride):
                    cur_idx = i * self.len_segment + j + cursor_batch * self.seq_stride
                    # print(f"i:{i}, j:{j}, self.len_segment:{self.len_segment}, cursor_batch:{cursor_batch}, self.seq_stride:{self.seq_stride}, cur_idx:{cur_idx}")
                    yield cur_idx
            cursor_batch += 1

    def __len__(self):
        assert False
        # return len(self.data)
        # return len(self.data_source)


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

    # TODO: remove before ?????

    dataset_len = len(dataset)
    logging.debug(
        f"Length of the dataset passed to generate_label_distribution_and_lds: {dataset_len}")
    logging.debug(f"kernel_size: {kernel_size}")
    logging.debug(f"kernel_std: {kernel_std}")
    logging.debug(f"Generating empirical distribution...")

    tab = np.array([dataset[i][3].item() for i in range(dataset_len)])
    tab = np.around(tab, decimals=5)
    elts = np.unique(tab)
    logging.debug(f"all labels: {elts}")
    dist, bins = np.histogram(
        tab, bins=nb_bins, density=False, range=(0.0, 1.0))

    # dist, bins = np.histogram([dataset[i][3].item() for i in range(dataset_len)], bins=nb_bins, density=False, range=(0.0, 1.0))

    logging.debug(f"dist: {dist}")

    # kernel = get_lds_kernel(kernel_size, kernel_std)
    # lds = convolve1d(dist, weights=kernel, mode='constant')

    lds = gaussian_filter1d(input=dist, sigma=kernel_std, axis=- 1,
                            order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

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
        logging.debug(
            f"The LDS distribution has {self.nb_bins} bins of width {self.bin_width}")
        self.weights = torch.tensor(self.weights)

        logging.debug(f"self.weights: {self.weights}")

    def lds_weights_batch(self, batch_labels):
        device = batch_labels.device
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        last_bin = 1.0 - self.bin_width
        batch_idxs = torch.minimum(batch_labels, torch.ones_like(
            batch_labels) * last_bin) / self.bin_width  # FIXME : double check
        batch_idxs = batch_idxs.floor().long()
        res = 1.0 / self.weights[batch_idxs]
        return res

    def __str__(self):
        return f"LDS nb_bins: {self.nb_bins}\nbins: {self.bins}\ndistribution: {self.distribution}\nlds_distribution: {self.lds_distribution}\nweights: {self.weights} "


def generate_dataloader(path_dataset, phase, test, window_size, fe, seq_len, seq_stride, distribution_mode, batch_size, nb_batch_per_epoch, classification, split_i,
                        network_stride):
    all_subject = pd.read_csv(
        Path(path_dataset) / subject_list, header=None, delim_whitespace=True).to_numpy()
    test_subject = None
    if phase == 'full':
        p1_subject = pd.read_csv(Path(
            path_dataset) / subject_list_p1, header=None, delim_whitespace=True).to_numpy()
        p2_subject = pd.read_csv(Path(
            path_dataset) / subject_list_p2, header=None, delim_whitespace=True).to_numpy()
        train_subject_p1, validation_subject_p1 = train_test_split(
            p1_subject, train_size=0.8, random_state=split_i)
        if test:
            test_subject_p1, validation_subject_p1 = train_test_split(
                validation_subject_p1, train_size=0.5, random_state=split_i)
        train_subject_p2, validation_subject_p2 = train_test_split(
            p2_subject, train_size=0.8, random_state=split_i)
        if test:
            test_subject_p2, validation_subject_p2 = train_test_split(
                validation_subject_p2, train_size=0.5, random_state=split_i)
        train_subject = np.array(
            [s for s in all_subject if s[0] in train_subject_p1[:, 0] or s[0] in train_subject_p2[:, 0]]).squeeze()
        if test:
            test_subject = np.array(
                [s for s in all_subject if s[0] in test_subject_p1[:, 0] or s[0] in test_subject_p2[:, 0]]).squeeze()
        validation_subject = np.array(
            [s for s in all_subject if s[0] in validation_subject_p1[:, 0] or s[0] in validation_subject_p2[:, 0]]).squeeze()
    else:
        train_subject, validation_subject = train_test_split(
            all_subject, train_size=0.8, random_state=split_i)
        if test:
            test_subject, validation_subject = train_test_split(
                validation_subject, train_size=0.5, random_state=split_i)
    logging.debug(f"Subjects in training : {train_subject[:, 0]}")
    logging.debug(f"Subjects in validation : {validation_subject[:, 0]}")
    if test:
        logging.debug(f"Subjects in test : {test_subject[:, 0]}")

    len_segment = LEN_SEGMENT * fe
    train_loader = None
    validation_loader = None
    test_loader = None
    batch_size_validation = None
    batch_size_test = None
    filename = filename_classification_dataset if classification else filename_regression_dataset

    if seq_len is not None:
        nb_segment_validation = len(
            np.hstack([range(int(s[1]), int(s[2])) for s in validation_subject]))
        batch_size_validation = len(list(range(
            0, (seq_stride // network_stride) * network_stride, network_stride))) * nb_segment_validation

        ds_train = SignalDataset(filename=filename,
                                 path=path_dataset,
                                 window_size=window_size,
                                 fe=fe,
                                 seq_len=seq_len,
                                 seq_stride=seq_stride,
                                 list_subject=train_subject,
                                 len_segment=len_segment)

        ds_validation = SignalDataset(filename=filename,
                                      path=path_dataset,
                                      window_size=window_size,
                                      fe=fe,
                                      seq_len=1,
                                      seq_stride=1,  # just to be sure, fixed value
                                      list_subject=validation_subject,
                                      len_segment=len_segment)
        idx_true, idx_false = get_class_idxs(ds_train, distribution_mode)
        samp_train = RandomSampler(idx_true=idx_true,
                                   idx_false=idx_false,
                                   batch_size=batch_size,
                                   nb_batch=nb_batch_per_epoch,
                                   distribution_mode=distribution_mode)

        samp_validation = ValidationSampler(ds_validation,
                                            seq_stride=seq_stride,
                                            len_segment=len_segment,
                                            nb_segment=nb_segment_validation,
                                            network_stride=network_stride)
        train_loader = DataLoader(ds_train,
                                  batch_size=batch_size,
                                  sampler=samp_train,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)

        validation_loader = DataLoader(ds_validation,
                                       batch_size=batch_size_validation,
                                       sampler=samp_validation,
                                       num_workers=0,
                                       pin_memory=True,
                                       shuffle=False)
    else:
        nb_segment_test = len(
            np.hstack([range(int(s[1]), int(s[2])) for s in test_subject]))
        batch_size_test = len(list(range(
            0, (seq_stride // network_stride) * network_stride, network_stride))) * nb_segment_test

        ds_test = SignalDataset(filename=filename,
                                path=path_dataset,
                                window_size=window_size,
                                fe=fe,
                                seq_len=1,
                                seq_stride=1,  # just to be sure, fixed value
                                list_subject=test_subject,
                                len_segment=len_segment)

        samp_test = ValidationSampler(ds_test,
                                      seq_stride=seq_stride,
                                      len_segment=len_segment,
                                      nb_segment=nb_segment_test,
                                      network_stride=network_stride)

        test_loader = DataLoader(ds_test,
                                 batch_size=batch_size_test,
                                 sampler=samp_test,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False)

    return train_loader, validation_loader, batch_size_validation, test_loader, batch_size_test, test_subject


def generate_dataloader_unlabelled_offline(unlabelled_segment,
                                           window_size,
                                           seq_stride,
                                           network_stride):
    nb_segment_test = 1
    batch_size_test = len(list(range(0, (seq_stride // network_stride)
                          * network_stride, network_stride))) * nb_segment_test
    unlabelled_segment = torch.tensor(
        unlabelled_segment, dtype=torch.float).squeeze()
    assert len(
        unlabelled_segment.shape) == 1, f"Segment has more than one dimension: {unlabelled_segment.shape}"
    len_segment = len(unlabelled_segment)
    ds_test = UnlabelledSignalDatasetSingleSegment(unlabelled_segment=unlabelled_segment,
                                                   window_size=window_size)
    samp_test = ValidationSampler(ds_test,
                                  seq_stride=seq_stride,
                                  # because we don't have additional data at the end on the signal here
                                  len_segment=len_segment-window_size,
                                  nb_segment=nb_segment_test,
                                  network_stride=network_stride)

    test_loader = DataLoader(ds_test,
                             batch_size=batch_size_test,
                             sampler=samp_test,
                             num_workers=0,
                             pin_memory=True,
                             shuffle=False)

    return test_loader, batch_size_test
