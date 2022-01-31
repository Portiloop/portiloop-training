import torch
import os
import wandb
import copy
from scipy.ndimage import gaussian_filter1d


class LoggerWandb:
    def __init__(self, experiment_name, c_dict, project_name):
        self.best_model = None
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=project_name, entity="portiloop", id=experiment_name, resume="allow",
                                    config=c_dict, reinit=True)

    def log(self,
            accuracy_train,
            loss_train,
            accuracy_validation,
            loss_validation,
            f1_validation,
            precision_validation,
            recall_validation,
            best_epoch,
            best_model,
            loss_early_stopping,
            best_epoch_early_stopping,
            best_model_accuracy_validation,
            best_model_f1_score_validation,
            best_model_precision_validation,
            best_model_recall_validation,
            best_model_loss_validation,
            best_model_on_loss_accuracy_validation,
            best_model_on_loss_f1_score_validation,
            best_model_on_loss_precision_validation,
            best_model_on_loss_recall_validation,
            best_model_on_loss_loss_validation,
            updated_model=False,
            ):
        self.best_model = best_model
        self.wandb_run.log({
            "accuracy_train": accuracy_train,
            "loss_train": loss_train,
            "accuracy_validation": accuracy_validation,
            "loss_validation": loss_validation,
            "f1_validation": f1_validation,
            "precision_validation": precision_validation,
            "recall_validation": recall_validation,
            "loss_early_stopping": loss_early_stopping,
        })
        self.wandb_run.summary["best_epoch"] = best_epoch
        self.wandb_run.summary["best_epoch_early_stopping"] = best_epoch_early_stopping
        self.wandb_run.summary["best_model_f1_score_validation"] = best_model_f1_score_validation
        self.wandb_run.summary["best_model_precision_validation"] = best_model_precision_validation
        self.wandb_run.summary["best_model_recall_validation"] = best_model_recall_validation
        self.wandb_run.summary["best_model_loss_validation"] = best_model_loss_validation
        self.wandb_run.summary["best_model_accuracy_validation"] = best_model_accuracy_validation
        self.wandb_run.summary["best_model_on_loss_f1_score_validation"] = best_model_on_loss_f1_score_validation
        self.wandb_run.summary["best_model_on_loss_precision_validation"] = best_model_on_loss_precision_validation
        self.wandb_run.summary["best_model_on_loss_recall_validation"] = best_model_on_loss_recall_validation
        self.wandb_run.summary["best_model_on_loss_loss_validation"] = best_model_on_loss_loss_validation
        self.wandb_run.summary["best_model_on_loss_accuracy_validation"] = best_model_on_loss_accuracy_validation
        if updated_model:
            self.wandb_run.save(os.path.join(
                path_dataset, self.experiment_name), policy="live", base_path=path_dataset)
            self.wandb_run.save(os.path.join(
                path_dataset, self.experiment_name + "_on_loss"), policy="live", base_path=path_dataset)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self, classif):
        if classif:
            self.wandb_run.restore(self.experiment_name, root=path_dataset)
        else:
            self.wandb_run.restore(
                self.experiment_name + "_on_loss", root=path_dataset)


def f1_loss(output, batch_labels):
    # logging.debug(f"output in loss : {output[:,1]}")
    # logging.debug(f"batch_labels in loss : {batch_labels}")
    y_pred = output
    tp = (batch_labels * y_pred).sum().to(torch.float32)
    tn = ((1 - batch_labels) * (1 - y_pred)).sum().to(torch.float32).item()
    fp = ((1 - batch_labels) * y_pred).sum().to(torch.float32)
    fn = (batch_labels * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    F1_class1 = 2 * tp / (2 * tp + fp + fn + epsilon)
    F1_class0 = 2 * tn / (2 * tn + fn + fp + epsilon)
    New_F1 = (F1_class1 + F1_class0) / 2
    return 1 - New_F1


def get_metrics(tp, fp, fn):
    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall


def get_lds_kernel(ks, sigma):
    half_ks = (ks - 1) // 2
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    kernel_window = gaussian_filter1d(
        base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    return kernel_window


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
        logging.debug(
            f"The SR distribution has {self.nb_bins} bins of width {self.bin_width}")
        logging.debug(f"Initial self.weights: {self.weights}")

    def update_and_get_weighted_loss(self, batch_labels, unweighted_loss):
        device = batch_labels.device
        if self.weights.device != device:
            logging.debug(f"Moving SR weights to {device}")
            self.weights = self.weights.to(device)
        last_bin = 1.0 - self.bin_width
        batch_idxs = torch.minimum(batch_labels, torch.ones_like(
            batch_labels) * last_bin) / self.bin_width  # FIXME : double check
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
            sum_mean_loss = torch.sum(
                mean_loss_per_idx_normalized[idx_changed])
            mean_loss_per_idx_normalized[idx_changed] = mean_loss_per_idx_normalized[idx_changed] * \
                sum_changed_weights / sum_mean_loss
            # logging.debug(f"old self.weights: {self.weights}")
            self.weights[idx_changed] = (1.0 - self.alpha) * self.weights[idx_changed] + \
                self.alpha * mean_loss_per_idx_normalized[idx_changed]
            self.weights /= torch.sum(self.weights)  # force sum to 1
            # logging.debug(f"unique_idx: {unique_idx}")
            # logging.debug(f"new self.weights: {self.weights}")
            # logging.debug(f"new torch.sum(self.weights): {torch.sum(self.weights)}")
        return torch.sqrt(res * self.nb_bins)

    def __str__(self):
        return f"LDS nb_bins: {self.nb_bins}\nweights: {self.weights}"
