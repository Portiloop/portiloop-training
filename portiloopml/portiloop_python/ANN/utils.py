from pathlib import Path
import torch
import os
import numpy as np
from random import seed
import wandb
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, kaiserord, filtfilt
import matplotlib.pyplot as plt
from portiloopml.portiloop_python.ANN.data.mass_data import SleepStageDataset


class RMSScorer:
    def __init__(self, fs=250, lowcut=11, highcut=16):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.stopband_att = 60
        self.width = 0.5
        self.ntaps, _ = kaiserord(
            self.stopband_att, self.width / (0.5 * self.fs))
        self.atten = kaiser_atten(self.ntaps, self.width / (0.5 * self.fs))
        self.beta = kaiser_beta(self.atten)
        self.a = 1.0
        self.taps = firwin(self.ntaps, [self.lowcut, self.highcut], fs=fs,
                           pass_zero=False, window=('kaiser', self.beta), scale=False)
        self.padlen = 1250
        self.size_window = int(0.5 * self.fs)
        # Index compared to the detection window
        self.baseline_idx = int(-2 * self.fs)

    def get_score(self, candidate, filter=True):
        # Filter the signal
        if filter:
            filtered_signal = filtfilt(
                self.taps, self.a, candidate, padlen=self.padlen)
        else:
            filtered_signal = candidate

        # Get the baseline and the detection window for the RMS
        detect_index = len(candidate) // 2
        baseline = filtered_signal[detect_index +
                                   self.baseline_idx:detect_index + self.baseline_idx + self.size_window]
        detection = filtered_signal[detect_index:detect_index +
                                    self.size_window]

        # Calculate the RMS
        baseline_rms = np.sqrt(np.mean(np.square(baseline)))
        detection_rms = np.sqrt(np.mean(np.square(detection)))

        score = detection_rms / baseline_rms
        return score

    def filter(self, sequence):
        return filtfilt(self.taps, self.a, sequence, padlen=self.padlen)


class SpindlePlotter:
    def __init__(self, signal, spindle_labels, sleep_stage_labels=None, model_spindle_labels=None, min_label_time=0.4, freq=250):
        self.signal = signal
        self.spindle_labels = spindle_labels
        self.sleep_stage_labels = sleep_stage_labels
        self.interval = int(min_label_time * freq)
        self.freq = freq
        self.scorer = RMSScorer()
        self.spindle_indexes, self.rms_scores = self.find_spindle_indexes()
        if model_spindle_labels is not None:
            self.model_spindle_labels = model_spindle_labels
            self.model_spindle_indexes, self.model_rms_scores = self.find_spindle_indexes(
                model=True)
        else:
            self.model_spindle_indexes = None

    def find_spindle_indexes(self, model=False):
        if model:
            spindle_labels = self.model_spindle_labels
        else:
            spindle_labels = self.spindle_labels

        # Find the indexes where we have a 1 followed by a zero
        indexes = np.where((spindle_labels == 1) & (
            np.roll(spindle_labels, 1) == 0))[0]
        indexes = indexes[np.insert(
            np.diff(indexes) >= self.interval, 0, True)]

        # rms_scores = [self.scorer.get_score(self.signal[index - 1250:index + 1250]) for index in indexes if index - 1250 >= 0 and index + 1250 < len(self.signal)]
        rms_scores = []

        return indexes, rms_scores

    def plot(self, index, model=False, time_before=10, time_after=5, rms_score=False):
        """
        Plot the signal around the spindle at given index
        """

        if model:
            main_indexes = self.model_spindle_indexes
            secondary_indexes = self.spindle_indexes
            rms_scores = self.model_rms_scores
        else:
            main_indexes = self.spindle_indexes
            secondary_indexes = self.model_spindle_indexes
            rms_scores = self.rms_scores

        if rms_score:
            index_signal = main_indexes[index]
            score = self.scorer.get_score(
                self.signal[index_signal - 1250:index_signal + 1250])
            print(f"RMS score: {score}")

        spindle_start = main_indexes[index]
        plot_start = spindle_start - time_before * self.freq
        # + 1 second to make sure to have the spindle and time after
        plot_end = spindle_start + time_after * self.freq + 1 * self.freq
        plt.figure(figsize=(20, 10))

        eeg_signal = self.signal[plot_start:plot_end]
        plt.plot(eeg_signal)

        # Check if there are other spindles in the window
        for i, spindle_index in enumerate(main_indexes):
            if spindle_index > plot_start and spindle_index < plot_end and i != index:
                plt.axvline(x=spindle_index - plot_start,
                            color='orange', label="Other primary spindles")

        # Check if there are spindles from our model in the window
        if self.model_spindle_indexes is not None:
            for i, spindle_index in enumerate(secondary_indexes):
                if spindle_index > plot_start and spindle_index < plot_end:
                    plt.axvline(x=spindle_index - plot_start,
                                color='blue', label="Secondary spindles")

        plt.axvline(x=time_before * self.freq,
                    color='r', label="Spindle Start")
        plt.xlabel("Index")
        plt.ylabel("Signal")

        # Add a legend
        plt.legend()

        plt.show()

        # sleep_stages = np.unique(self.sleep_stage_labels[spindle_start])
        # if len(sleep_stages) == 1:
        #     print("Sleep stage: ", SleepStageDataset.get_labels()[sleep_stages[0]])
        # else:
        #     print("Sleep stages: ", [SleepStageDataset.get_labels()[sleep_stage] for sleep_stage in sleep_stages])

    def num_spindles_labels(self):
        return len(self.spindle_indexes)

    def num_spindles_model(self):
        if self.model_spindle_indexes is not None:
            return len(self.model_spindle_indexes)
        else:
            return 0


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
            self.wandb_run.restore(self.experiment_name,
                                   root=self.c_dict['path_dataset'])
        else:
            self.wandb_run.restore(
                self.experiment_name + "_on_loss", root=self.c_dict['path_dataset'])


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


def get_metrics(predictions, labels):
    """
    Compute the F1, precision and recall for spindles from a true positive count, false positive count and false negative count
    """
    # n_classes = len(torch.unique(labels))

    # tp = torch.zeros(n_classes)
    # tn = torch.zeros(n_classes)
    # fp = torch.zeros(n_classes)
    # fn = torch.zeros(n_classes)

    # for i in range(n_classes):
    #     tp[i] = torch.sum((predictions == i) & (labels == i))
    #     tn[i] = torch.sum((predictions != i) & (labels != i))
    #     fp[i] = torch.sum((predictions == i) & (labels != i))
    #     fn[i] = torch.sum((predictions != i) & (labels == i))

    # epsilon = 1e-7

    # accuracy = torch.sum(tp) / torch.sum(tp + tn + fp + fn)
    # recall = torch.mean(tp / (tp + fn + epsilon))
    # precision = torch.mean(tp / (tp + fp + epsilon))
    # f1_score = torch.mean(2 * precision * recall / (precision + recall + epsilon))

    # return accuracy, recall, precision, f1_score
    acc = (predictions == labels).float().mean()
    predictions = predictions.float()
    labels = labels.float()

    # Get the true positives, true negatives, false positives and false negatives
    tp = (labels * predictions)
    tn = ((1 - labels) * (1 - predictions))
    fp = ((1 - labels) * predictions)
    fn = (labels * (1 - predictions))

    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return acc, f1, precision, recall


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
        'old_dataset': Path(__file__).absolute().parent.parent.parent / 'dataset',
        'MASS_dir': Path(__file__).absolute().parent.parent.parent / 'dataset' / 'MASS',
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
        'batch_size': 64,
        'dropout': 0.5,
        'adam_w': 0.01,
        'reg_balancing': 'none',
        'lr_adam': 0.0005,

        # Stopping parameters
        'nb_epoch_max': 1500,
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
        'nb_channel': 31,
        'in_channels': 1,

        # Attention stuff
        'max_h_length': 50,  # How many time steps to consider in the attention
        'n_heads': 4,  # How many attention heads to use
        'after_rnn': None,  # Whether to put attention, CNN, or nothing after the RNN

        # IDK
        'time_in_past': 8.5,
        'estimator_size_memory': 188006400,

        # Device info
        'device_train': 'cuda',
        'device_val': 'cuda',
        'device_inference': 'cpu',

        # Data info
        'fe': 250,
        'validation_network_stride': 10,
        'phase': "full",
        'split_idx': 0,
        'threshold': 0.5,
        'window_size': 54,
        'seq_stride': 42,
        'nb_batch_per_epoch': 1000,
        'distribution_mode': 0,
        'seq_len': 50,
        'seq_stride_s': 0.170,
        'window_size_s': 0.218,
        'len_segment_s': 115,
        'out_features': 1,  # Number of output features
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
