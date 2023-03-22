from pathlib import Path
import torch
import os
import numpy as np
from random import seed
import wandb


class LoggerWandb:
    def __init__(self, experiment_name, c_dict, project_name, group=None):
        self.best_model = None
        if group is None:
            group = "default"
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=project_name, entity="portiloop", id=experiment_name, resume="allow",
                                    config=c_dict, reinit=True, group=group)
        self.c_dict = c_dict            

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
        # if updated_model:
        #     self.wandb_run.save(os.path.join(path_dataset, self.experiment_name), policy="live", base_path=path_dataset)
        #     self.wandb_run.save(os.path.join(path_dataset, self.experiment_name + "_on_loss"), policy="live", base_path=path_dataset)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self, classif):
        if classif:
            self.wandb_run.restore(self.experiment_name, root=self.c_dict['path_dataset'])
        else:
            self.wandb_run.restore(self.experiment_name + "_on_loss", root=self.c_dict['path_dataset'])

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
    """
    Compute the F1, precision and recall for spindles from a true positive count, false positive count and false negative count
    """
    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall


def get_final_model_config_dict(index=0, split_i=0):
    """
    Configuration dictionary of the final 1-input pre-trained model presented in the Portiloop paper.

    Args:
        index: last number in the name of the pre-trained model (several are provided)
        split_i: index of the random train/validation/test split (you can ignore this for inference)

    Returns:
        configuration dictionary of the pre-trained model
    """
    c_dict = {'experiment_name': f'sanity_check_final_model_3',
              'device_train': 'cuda',
              'device_val': 'cuda',
              'device_inference': 'cpu',
              'nb_epoch_max': 150,
              'max_duration': 257400,
              'nb_epoch_early_stopping_stop': 100,
              'early_stopping_smoothing_factor': 0.1,
              'fe': 250,
              'nb_batch_per_epoch': 1000,
              'first_layer_dropout': False,
              'power_features_input': False,
              'dropout': 0.5,
              'adam_w': 0.01,
              'distribution_mode': 0,
              'classification': True,
              'reg_balancing': 'none',
              'split_idx': split_i,
              'validation_network_stride': 1,
              'nb_conv_layers': 3,
              'seq_len': 50,
              'nb_channel': 31,
              'hidden_size': 7,
              'seq_stride_s': 0.170,
              'nb_rnn_layers': 1,
              'RNN': True,
              'envelope_input': False,
              'lr_adam': 0.0005,
              'batch_size': 256,
              'stride_pool': 1,
              'stride_conv': 1,
              'kernel_conv': 7,
              'kernel_pool': 7,
              'dilation_conv': 1,
              'dilation_pool': 1,
              'nb_out': 18,
              'time_in_past': 8.5,
              'estimator_size_memory': 188006400}
    return c_dict

def get_configs(exp_name, test_set, seed_exp):
    """
    Get the configuration dictionaries containgin information about:
        - Paths where data is stored
        - Model info
        - Data info
    """

    config = {
        # Path info'
        'old_dataset': Path("/project/portiloop_transformer/transformiloop/dataset"),
        'MASS_dir': Path("/project/portiloop_transformer/transformiloop/dataset/MASS_preds"),
        'path_models': Path(__file__).absolute().parent.parent.parent / 'models',
        'path_dataset': Path(__file__).absolute().parent.parent.parent / 'dataset',
        'filename_regression_dataset': f"dataset_regression_full_big_250_matlab_standardized_envelope_pf.txt",
        'filename_classification_dataset': f"dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt",
        'subject_list': f"subject_sequence_full_big.txt",
        'subject_list_p1': f"subject_sequence_p1_big.txt",
        'subject_list_p2': f"subject_sequence_p2_big.txt",

        # Experiment info
        'experiment_name': exp_name,
        'seed_exp': seed_exp,
        'test_set': test_set,

        # Training hyperparameters
        'batch_size': 256,
        'dropout': 0.5,
        'adam_w': 0.01,
        'reg_balancing': 'none',
        'lr_adam': 0.0005,

        # Stopping parameters
        'nb_epoch_max': 150,
        'max_duration': 257400,
        'nb_epoch_early_stopping_stop': 100,
        'early_stopping_smoothing_factor': 0.1,

        # Model info
        'first_layer_dropout': False,
        'power_features_input': False,
        'RNN': True,
        'envelope_input': False,
        'classification': True,

        # CNN stuff
        'nb_conv_layers': 3,
        'nb_channel': 31,
        'stride_pool': 1,
        'stride_conv': 1,
        'kernel_conv': 7,
        'kernel_pool': 7,
        'dilation_conv': 1,
        'dilation_pool': 1,

        # RNN stuff
        'nb_rnn_layers': 1,
        'nb_out': 18,
        'hidden_size': 7,

        # Attention stuff
        'max_h_length': 50, # How many time steps to consider in the attention
        'n_heads': 4, # How many attention heads to use
        'after_rnn': None, # Whether to put attention, CNN, or nothing after the RNN

        # IDK
        'time_in_past': 8.5,
        'estimator_size_memory': 188006400,

        # Device info
        'device_train': 'cuda',
        'device_val': 'cuda',
        'device_inference': 'cpu',

        # Data info
        'fe': 250,
        'validation_network_stride': 1,
        'phase': "full",
        'split_idx': 0,
        'threshold': 0.5,
        'window_size': 54,
        'seq_stride': 42,
        'nb_batch_per_epoch': 10000,
        'distribution_mode': 0,
        'seq_len': 50,
        'seq_stride_s': 0.170,
        'window_size_s': 0.218,
        'len_segment_s': 115,
    }

    return config


def set_seeds(seed_int):
    """
    Set seeds for reproducibility (see https://github.com/pranshu28/TAG)
    """
    os.environ['PYTHONHASHSEED'] = str(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed(seed_int)
    torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(seed_int)


def get_device():
    """
    Get device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


