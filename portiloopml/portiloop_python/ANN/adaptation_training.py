import argparse
from collections import deque
import json
import os
import random
import time
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
from torch import nn
from torch import optim
import copy
import numpy as np
from tqdm import tqdm
from portiloopml.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, SleepStageDataset, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloopml.portiloop_python.ANN.data.mass_data_new import MassConsecutiveSampler, MassDataset, SubjectLoader
from portiloopml.portiloop_python.ANN.lightning_tests import load_model
from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork, get_trained_model
from portiloopml.portiloop_python.ANN.utils import get_configs, get_metrics, set_seeds
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, kaiserord, filtfilt
from portiloopml.portiloop_python.ANN.validation_mass import load_model_mass

from portiloopml.portiloop_python.ANN.wamsley_utils import binary_f1_score, detect_lacourse, detect_wamsley, get_spindle_onsets, plot_spindle


class AdaptationSampler2(torch.utils.data.Sampler):
    def __init__(self, spindle_indexes, non_spindle_range, past_signal_len, window_size, replay_dataset=None, replay_multiplier=1):
        """
        Sample random items from a dataset
        """
        self.spindle_indexes = spindle_indexes
        self.non_spindle_range = non_spindle_range
        self.past_signal_len = past_signal_len
        self.window_size = window_size
        self.replay_dataset = replay_dataset
        self.replay_multiplier = replay_multiplier

    def __iter__(self):
        """
        Returns an iterator over the dataset, and the replay dataset
        """
        # Choose the same number of spindles and non spindles
        num_spindles = len(self.spindle_indexes)
        non_spindle_indexes = random.sample(
            range(self.non_spindle_range[0], self.non_spindle_range[1]), num_spindles)

        # Add the labels to the indexes for the new found spindles
        self.spindle_indexes = [
            (i, 1, False) for i in self.spindle_indexes]
        non_spindle_indexes = [
            (i, 0, False) for i in non_spindle_indexes]
        self.indexes = self.spindle_indexes + non_spindle_indexes

        # Add the replay dataset indexes:
        if self.replay_dataset is not None:
            num_replay_sample = int(self.replay_multiplier * num_spindles)
            # num_replay_sample = int((1000 * 64) / 2)
            replay_spindle_indexes = random.sample(
                self.replay_dataset.labels_indexes['spindle_label'].tolist(), num_replay_sample)
            replay_non_spindle_indexes = random.sample(
                np.arange(self.replay_dataset.past_signal_len, len(self.replay_dataset)).tolist(), num_replay_sample)
            replay_non_spindle_indexes = [
                (index, 0, True) for index in replay_non_spindle_indexes]
            replay_spindle_indexes = [
                (index, 1, True) for index in replay_spindle_indexes]
            self.indexes = replay_spindle_indexes + replay_non_spindle_indexes

        random.shuffle(self.indexes)

        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class AdaptationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, weights):
        """
        Sample random items from a dataset
        """
        self.dataset = dataset
        self.stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }
        self.weights = weights

    def __iter__(self):
        """
        Returns an iterator over the dataset
        """
        while True:
            # Choose between one of the 4 types of samples
            possibilities = [0, 1, 2, 3]

            # Sample from the possibilities with the weights
            sample_type = random.choices(possibilities, self.weights)[0]

            if sample_type == 0 or sample_type == 1:
                # We pick a spindle
                if len(self.dataset.sampleable_spindles) > 0:
                    index = random.choice(self.dataset.sampleable_spindles)
                    # We remove the index from the sampleable list
                    # self.dataset.sampleable_spindles.remove(index)
                    label = 1
                else:
                    index = -1
            else:
                # We pick a non spindle
                index = random.randint(
                    0, len(self.dataset.wamsley_buffer) - 1)
                label = 0

            # if sample_type == 0:
            #     # We pick a false negative
            #     if len(self.dataset.sampleable_missed_spindles) > 0:
            #         index = random.choice(
            #             self.dataset.sampleable_missed_spindles)
            #         self.stats['false_negatives'] += 1
            #         # We remove the index from the sampleable list
            #         self.dataset.sampleable_missed_spindles.remove(index)
            #         label = 0
            #     else:
            #         index = -1
            # elif sample_type == 1:
            #     # We pick a true positive
            #     if len(self.dataset.sampleable_found_spindles) > 0:
            #         index = random.choice(
            #             self.dataset.sampleable_found_spindles)
            #         # We remove the index from the sampleable list
            #         self.stats['true_positives'] += 1
            #         # We remove the index from the sampleable list
            #         self.dataset.sampleable_found_spindles.remove(index)
            #         label = 1
            #     else:
            #         index = -1
            # elif sample_type == 2:
            #     # We pick a false positive
            #     if len(self.dataset.sampleable_false_spindles) > 0:
            #         index = random.choice(
            #             self.dataset.sampleable_false_spindles)
            #         self.stats['false_positives'] += 1
            #         # We remove the index from the sampleable list
            #         self.dataset.sampleable_false_spindles.remove(index)
            #         label = 2
            #     else:
            #         index = -1
            # else:
            #     # We pick a true negative
            #     index = random.randint(
            #         0, len(self.dataset.wamsley_buffer) - 1)
            #     self.stats['true_negatives'] += 1
            #     label = 3

            # Check if the index is far enough from the begginins of the buffer
            if index < self.dataset.min_signal_len:
                continue

            yield index - self.dataset.past_signal_len - self.dataset.window_size + 1, label


class AdaptationDataset(torch.utils.data.Dataset):
    def __init__(self, config, batch_size):
        """
        Store items from a dataset 
        """
        self.real_index = 0  # Used to keep track of the real index of the sample in the buffer
        # Used to keep track of all the spindles found by online wamsley
        self.total_spindles = []
        self.wamsley_thresholds = []
        # Arbitrarily chose 10 minutes of signal to be the interval on which we run Wamsley
        self.wamsley_interval = 250 * 60 * config['adaptation_interval']
        self.batch_size = batch_size
        self.samples = []
        self.window_buffer = []
        self.spindle_indexes = []
        self.non_spindle_indexes = []
        self.pred_threshold = 0.50

        self.use_mask_wamsley = config['use_mask_wamsley']

        # signal needed before the last window
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        self.window_size = config['window_size']
        self.past_signal_len = (self.seq_len - 1) * \
            self.seq_stride
        self.min_signal_len = self.past_signal_len + self.window_size

        # Buffers
        self.wamsley_buffer = []
        self.ss_pred_buffer = []
        self.ss_label_buffer = []
        self.prediction_buffer = []
        self.prediction_buffer_raw = []
        self.spindle_labels = []
        self.last_wamsley_run = 0
        self.online_wamsley_spindles = []

        self.wamsley_outs = []

        # Sampleable lists
        # Indexes of false negatives that can be sampled
        self.sampleable_missed_spindles = []
        # Indexes of false positives that can be sampled
        self.sampleable_false_spindles = []
        # Indexes of true positives that can be sampled
        self.sampleable_found_spindles = []
        # INdexes of all the spindles that can be sampled
        self.sampleable_spindles = []

        # We assume that the true negatives are just all the rest given that they are majority
        self.num_true_positive = []
        self.num_false_positive = []
        self.num_false_negative = []

        # Used for tests:
        self.used_thresholds = []
        self.candidate_thresholds = np.arange(
            config['min_threshold'], config['max_threshold'], 0.01)
        self.adapt_threshold_detect = config['adapt_threshold_detect']
        self.adapt_threshold_wamsley = config['adapt_threshold_wamsley']
        self.learn_wamsley = config['learn_wamsley']
        self.use_ss_label = config['use_ss_label']
        wamsley_config = config['wamsley_config']
        # self.wamsley_func = lambda x, y, z: detect_wamsley(
        #     x,
        #     y,
        #     thresholds=z,
        #     fixed=wamsley_config['fixed'],
        #     squarred=wamsley_config['squarred'],
        #     remove_outliers=wamsley_config['remove_outliers'],
        #     threshold_multiplier=wamsley_config['threshold_multiplier'],
        #     sampling_rate=wamsley_config['sampling_rate'],
        # )

        self.wamsley_func = lambda x, y, z: (detect_lacourse(
            x,
            y,
            sampling_rate=wamsley_config['sampling_rate']
        ), None, None, None, None)

        if config['use_replay']:
            self.replay_dataset = MassDataset(
                dataset_path,
                subjects=config['replay_subjects'],
                window_size=config['window_size'],
                seq_stride=config['seq_stride'],
                seq_len=50,
                use_filtered=False,
                compute_spindle_labels=False,
                wamsley_config=config['wamsley_config']
            )
        else:
            self.replay_dataset = None

    def __getitem__(self, index):
        """
        Returns a sample from the dataset
        """
        # Get the index and the sample type
        index, label, replay = index

        if replay:
            # Get the data from the replay dataset
            signal, _ = self.replay_dataset[index]
            category = label
            label = torch.Tensor([label]).type(torch.LongTensor)
            return signal, label, category

        # Get data
        signal = torch.Tensor(
            self.wamsley_buffer[index - self.past_signal_len:index + self.window_size])

        signal = signal.unfold(
            0, self.window_size, self.seq_stride)
        signal = signal.unsqueeze(1)

        # if label == 1:
        #     around_spindle = self.wamsley_buffer[index - 250:index + 250]
        #     spindle_on_off = self.spindle_labels[index - 250:index + 250]
        #     plot_spindle(around_spindle, spindle_on_off)
        #     plt.clf()
        #     plt.plot(signal[-1, 0, :])
        #     plt.savefig(f"spindle_signal_sampled.png")
        #     print()

        # Make sure that the last index of the signal is the same as the label
        # assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        # Keep the categorical loss to see if it improves in all domains or only some of them

        assert signal.shape == (
            self.seq_len, 1, self.window_size), "Signal shape is not correct"

        category = label
        if label == 0 or label == 2:
            label = torch.Tensor([0]).type(torch.LongTensor)
        else:
            label = torch.Tensor([1]).type(torch.LongTensor)

        return signal, label, category

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.wamsley_buffer) - self.min_signal_len

    def get_train_val_split(self, split=0.8):
        """
        Selects a split of the available data for training
        """
        total_indexes_pos = len(self.sampleable_spindles)
        split_index_pos = int(total_indexes_pos * split)

        # Keep only the indexes which are sampleable
        sampleable_indexes = np.array(self.sampleable_spindles)
        sampleable_indexes = sampleable_indexes[(sampleable_indexes >= self.past_signal_len) & (
            sampleable_indexes <= len(self.wamsley_buffer) - self.window_size)]
        train_spindle_indexes = sampleable_indexes[:split_index_pos]
        val_spindle_indexes = sampleable_indexes[split_index_pos:]

        total_indexes_neg = len(self.wamsley_buffer) - self.min_signal_len
        split_index_neg = int(total_indexes_neg * split)
        train_indexes = (self.past_signal_len,
                         split_index_neg - self.window_size)
        val_indexes = (split_index_neg + self.past_signal_len,
                       total_indexes_neg - self.window_size)

        return train_indexes, val_indexes, train_spindle_indexes, val_spindle_indexes

    def add_window(self, window, ss_pred, ss_label, spindle_pred, spindle_label, detect_threshold, force_wamsley=False):
        """
        Adds a window to the buffers to make it a sample when enough windows have arrived

        Args:
        window: The window to add
        ss_label: The sleep stage label
        model_pred: The model prediction
        spindle_label: The spindle label
        detect_threshold: The detection threshold
        force_wamsley: If we want to force Wamsley to run

        Returns:
        A boolean indicating if Wamsley was run
        """
        # Prepare the data to be added to the buffers
        points_to_add = window.squeeze().tolist() if len(
            self.wamsley_buffer) == 0 else window.squeeze().tolist()[-self.seq_stride:]

        # Check the sleep stage label
        # if not self.use_mask_wamsley:
        #     ss_label = True
        #     ss_mask_to_add = [ss_label] * len(points_to_add)
        # elif type(ss_label) == int:
        #     if (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
        #         ss_label = True
        #     else:
        #         ss_label = False
        #     ss_mask_to_add = [ss_label] * len(points_to_add)
        # else:
        #     ss_label = ss_label.squeeze(0)
        #     ss_mask_to_add = ((ss_label == SleepStageDataset.get_labels().index(
        #         "2")) | (ss_label == SleepStageDataset.get_labels().index("3"))).tolist()
        #     ss_mask_to_add = ss_mask_to_add[-len(points_to_add):]

        preds_raw_to_add = [spindle_pred.cpu()] * len(points_to_add)
        preds_to_add = [bool(spindle_pred.cpu() >= detect_threshold)] * \
            len(points_to_add)
        spindle_label_to_add = [spindle_label] * len(points_to_add)
        ss_pred_to_add = [ss_pred] * len(points_to_add)
        ss_label_to_add = ss_label.squeeze(0)[-len(points_to_add):]

        # Add to the buffers
        self.wamsley_buffer += points_to_add
        self.ss_pred_buffer += ss_pred_to_add
        self.ss_label_buffer += ss_label_to_add
        self.prediction_buffer += preds_to_add
        self.prediction_buffer_raw += preds_raw_to_add
        self.spindle_labels += spindle_label_to_add

        # Update the last wamsley run counter
        self.last_wamsley_run += len(points_to_add)

        assert len(self.ss_pred_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.prediction_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.spindle_labels) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.ss_label_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.prediction_buffer_raw) == len(
            self.wamsley_buffer), "Buffers are not the same length"

        if self.last_wamsley_run >= self.wamsley_interval or force_wamsley:
            self.run_spindle_detection(self.last_wamsley_run)
            self.last_wamsley_run = 0
            return True

        return False

    def run_spindle_detection(self, detect_len):

        # Run Wamsley on the buffer
        usable_buffer = self.wamsley_buffer[-detect_len:]
        if not self.use_mask_wamsley:
            usable_mask = [2] * len(usable_buffer)
        elif self.use_ss_label:
            usable_mask = self.ss_label_buffer[-detect_len:]
        else:
            usable_mask = self.ss_pred_buffer[-detect_len:]

        usable_mask = np.array(usable_mask)
        usable_mask = ((usable_mask == SleepStageDataset.get_labels().index(
            "2")) | (usable_mask == SleepStageDataset.get_labels().index("3")))
        usable_labels = self.spindle_labels[-detect_len:]

        if sum(usable_mask) > 0:
            wamsley_spindles, _, _, _, _ = self.wamsley_func(
                np.array(usable_buffer),
                np.array(usable_mask),
                self.wamsley_thresholds if self.adapt_threshold_wamsley else None
            )
        else:
            wamsley_spindles = []

        # Get the spindles we detected using Lacourse
        events_online = np.array(wamsley_spindles)
        if len(events_online) != 0:
            events_online_ranges = np.concatenate([np.arange(start, stop)
                                                   for start, stop in zip(events_online[:, 0], events_online[:, 2])])
            events_online_indexes = np.hstack(events_online_ranges)
        else:
            events_online_indexes = []

        # Get the spindles from the labels:
        event_labels_ranges = np.where(np.array(usable_labels) == 1)[0]
        if len(event_labels_ranges) != 0:
            event_labels_indexes = np.hstack(event_labels_ranges)
        else:
            event_labels_indexes = []

        # Get the difference in indexes between the buffer we studied and the total buffer
        index_diff = len(self.wamsley_buffer) - detect_len

        if self.learn_wamsley:
            self.sampleable_spindles += (np.array(events_online_indexes) +
                                         index_diff).tolist()
            new_info = len(events_online_indexes)
        else:
            self.sampleable_spindles += (np.array(event_labels_indexes) +
                                         index_diff).tolist()
            new_info = len(event_labels_indexes)

        return new_info

    def run_metrics(self):
        '''
        Get the metrics for the current predictions.

        This uses the predistions that were added with the used threshold in real time.
        '''
        # Compute the metrics on the entire buffer
        metrics = spindle_metrics(
            np.array(self.spindle_labels),
            np.array(self.prediction_buffer),
            threshold=0.5,
            sampling_rate=250,
            min_label_time=0.5)

        ss_metrics = staging_metrics(
            torch.tensor(self.ss_label_buffer),
            torch.tensor(self.ss_pred_buffer))

        return metrics, ss_metrics

    def get_thresholds(self):
        '''
        Returns the best threshold for the model compared to Wamsley threshold
        '''
        # Our predictions
        spindle_preds = torch.tensor(self.prediction_buffer_raw)
        spindle_labels = self.get_lacourse_spindle_vector()

        best_f1, best_thresh = hierarchical_search(
            lambda x: spindle_metrics(
                spindle_labels,
                spindle_preds,
                threshold=x,
                sampling_rate=250,
                min_label_time=0.5)['f1']
        )

        return best_thresh, best_f1

    def spindle_percentage(self):
        sum_spindles = sum([i[1] for i in self.samples if i[1] == 1])
        return sum_spindles / len(self)

    def has_samples(self):
        return len(self.sampleable_spindles) > self.batch_size
        # return len(self.sampleable_false_spindles) > self.batch_size\
        #     or len(self.sampleable_found_spindles) > self.batch_size \
        #     or len(self.sampleable_missed_spindles) > self.batch_size

    def get_lacourse_spindle_vector(self):
        """
        Returns the spindle vector
        """
        vector = np.zeros((len(self.wamsley_buffer)))
        vector[self.sampleable_spindles] = 1
        return vector


def run_adaptation(dataloader, val_dataloader, net, device, config, train, logger):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """

    # Run Wamsley to compare to the online Wamsley
    batch_size = config['batch_size']

    # Initialize adaptation dataset stuff
    adap_dataset = AdaptationDataset(config, batch_size)

    # Initialize All the necessary variables
    net_inference = copy.deepcopy(net)
    net_inference = net_inference.to(device)

    net_init = copy.deepcopy(net)
    net_init = net_init.to(device)
    net_init.eval()

    # optimizer = optim.SGD(net_copy.parameters(), lr=0.000003, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    training_losses = []
    inference_loss = []
    train_loss_tp = []
    train_loss_fn = []
    n = 0

    # Keep the sleep staging labels and predictions
    used_threshold = config['starting_threshold']
    spindle_pred_with_thresh = []
    used_lr = config['lr']

    last_n_ss = deque(maxlen=config['n_ss_smoothing'])

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    counter = 0
    for index, batch in enumerate(tqdm(dataloader)):

        net_inference.eval()
        with torch.no_grad():
            counter += 1

            # Get the data and labels
            window_data, labels = batch

            window_data = window_data.to(device)
            window_labels = labels['spindle_label'].to(device)

            # Get the output of the network
            spindle_output, ss_output, h1, _ = net_inference(window_data, h1)

            # Compute the loss
            output = spindle_output.squeeze(-1)
            window_labels = window_labels.float()

            # Update the loss
            inf_loss_step = criterion(output, window_labels).mean().item()
            inference_loss.append(inf_loss_step)

            logger.log({'adap_inference_loss': inf_loss_step}, step=index)

            # Add the window to the adaptation dataset given the sleep stage
            output = torch.sigmoid(output)
            ss_labels = labels['all_sleep_stage']

            if config['use_ss_smoothing']:
                # Use a voting system to smooth the sleep stage depending on the last n predictions
                this_output = torch.softmax(ss_output, dim=1)
                last_n_ss.append(this_output.cpu().detach().numpy())

                # Compute the average over the last n predictions
                nplast_n_ss = torch.tensor(last_n_ss)
                ss_output = torch.mean(nplast_n_ss, dim=0)
                ss_output = torch.argmax(ss_output)
            else:
                ss_output = torch.argmax(ss_output, dim=1)

            spindle_pred_with_thresh.append(
                output.cpu().item() > used_threshold)

            out_dataset = adap_dataset.add_window(
                window_data,
                ss_output,
                ss_labels,
                output,
                window_labels.cpu().item(),
                detect_threshold=used_threshold,
            )

        # If we have just added some new data, we log the metrics
        if out_dataset or index == len(dataloader) - 1:
            metrics, ss_metrics = adap_dataset.run_metrics()
            logger.log({'adap_online_f1': metrics['f1']}, step=index)
            logger.log(
                {'adap_online_precision': metrics['precision']}, step=index)
            logger.log({'adap_online_recall': metrics['recall']}, step=index)

            # Get the Confusion matrix
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            tn = len(adap_dataset.spindle_labels) - tp - fp - fn
            cm = np.array([[tn, fp], [fn, tp]])

            # Create a matplotlib figure for the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['non-spindle', 'spindle'])
            disp.plot()
            logger.log(
                {
                    'adap_online_cm': plt,
                },
                commit=False,
                step=index,
            )

            ss_metrics = ss_metrics[0]
            logger.log({'adap_ss_metrics': ss_metrics['accuracy']}, step=index)

        # Used for testing online Lacourse
        # if out_dataset:
        #     subject = '01-03-0025'
        #     baseline_signal = dataloader.dataset.data[subject]['signal'][:len(
        #         adap_dataset.wamsley_buffer)]
        #     baseline_mask = (dataloader.dataset.data[subject]['ss_label'][:len(adap_dataset.wamsley_buffer)] == 1) | (
        #         dataloader.dataset.data[subject]['ss_label'][:len(adap_dataset.wamsley_buffer)] == 2)
        #     baseline_spindles = detect_lacourse(
        #         np.array(baseline_signal), np.array(baseline_mask), sampling_rate=250)
        #     baseline_spindles = [e[0] for e in baseline_spindles]
        #     # small_signal = adap_dataset.wamsley_buffer
        #     # small_mask = adap_dataset.ss_pred_buffer
        #     # online_spindles = detect_lacourse(
        #     #     np.array(small_signal), np.array(small_mask), sampling_rate=250)
        #     # online_spindles_mask = detect_lacourse(
        #     #     np.array(small_signal), np.array(baseline_mask), sampling_rate=250)
        #     # online_spindles = [e[0] for e in online_spindles]
        #     # online_spindles_mask = [e[0] for e in online_spindles_mask]
        #     online_spindles = [i[0] for i in adap_dataset.total_spindles]

        #     metrics = binary_f1_score(
        #         np.array(baseline_spindles), np.array(online_spindles), 250)
        #     # metrics_real_mask = binary_f1_score(
        #     #     np.array(baseline_spindles), np.array(online_spindles_mask), 250)
        #     print(metrics)

        # If we have just added some new detection data, train is on and we have enough samples, we train
        if (out_dataset and train and adap_dataset.has_samples()):

            train_indexes, _, train_spindle_indexes, _ = adap_dataset.get_train_val_split(
                split=1.0)

            # Initialize the dataloaders
            train_sampler = AdaptationSampler2(
                train_spindle_indexes,
                train_indexes,
                window_size=adap_dataset.window_size,
                past_signal_len=adap_dataset.past_signal_len,
                replay_dataset=adap_dataset.replay_dataset,
                replay_multiplier=config['replay_multiplier'])

            train_dataloader = torch.utils.data.DataLoader(
                adap_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0)

            net_copy = copy.deepcopy(net_inference)
            optimizer = optim.Adam(net_copy.parameters(),
                                   lr=used_lr, weight_decay=config['adam_w'])
            fns = 0
            fps = 0
            # Training loop
            for batch, label, category in tqdm(train_dataloader):
                net_copy.train()
                # Training Loop
                train_sample = batch.to(device)
                train_label = label.to(device)

                optimizer.zero_grad()

                # Get the output of the network
                h_zero = torch.zeros((config['nb_rnn_layers'], train_sample.size(
                    0), config['hidden_size']), device=device)
                output, _, _, _ = net_copy(train_sample, h_zero)

                # Compute the loss
                output = output.squeeze(-1)
                train_label = train_label.squeeze(-1).float()
                loss_step = criterion(output, train_label)

                output_sig = torch.sigmoid(output)
                # Get the number of false positives and false negatives
                fp = sum((output_sig > used_threshold) & (
                    train_label == 0)).cpu().detach().numpy()
                fps += fp
                fn = sum((output_sig < used_threshold) & (
                    train_label == 1)).cpu().detach().numpy()
                fns += fn

                # Split the categories
                train_loss_fn.append(
                    loss_step[train_label == 0].mean().cpu().detach().numpy())
                train_loss_tp.append(
                    loss_step[train_label == 1].mean().cpu().detach().numpy())

                loss_step = loss_step.mean()
                loss_step.backward()
                optimizer.step()

                training_losses.append(loss_step.item())

            print(
                f"Trained on {fns} false negatives and {fps} false positives")

            # Average the weights of the model
            net_inference = average_weights(
                net_init, net_copy, alpha=config['alpha_training'])
            # net_inference = average_weights(
            #     net_inference, net_copy, alpha=config['alpha_training'])

            # Reset the hidden state of the GRU
            h1 = torch.zeros((config['nb_rnn_layers'], 1,
                              config['hidden_size']), device=device)

        if out_dataset:
            # Run the adaptation of threshold
            new_thresh, _ = adap_dataset.get_thresholds()
            logger.log({'best_threshold': new_thresh}, step=index)

            if config['adapt_threshold_detect']:
                used_threshold = new_thresh

        if out_dataset or index == 0 or index == len(dataloader) - 1:
            # Validation loop
            net_inference.eval()
            validation_losses_epoch = []
            val_loss_spindles_epoch = []
            val_loss_non_spindles_epoch = []
            val_labels = []
            val_preds = []
            with torch.no_grad():
                h_val = torch.zeros(
                    (config['nb_rnn_layers'], val_dataloader.batch_size, config['hidden_size']), device=device)
                for batch, label in tqdm(val_dataloader):
                    val_sample = batch.to(device)
                    val_label = label['spindle_label'].to(device)

                    # Get the output of the network
                    output, _, h_val, _ = net_inference(val_sample, h_val)

                    # Compute the loss
                    output = output.squeeze(-1)
                    val_label = val_label.squeeze(-1).float()
                    loss_step = criterion(output, val_label)

                    # Append to the lists
                    val_label = val_label.cpu().detach().numpy()
                    val_preds.append(torch.sigmoid(
                        output).cpu().detach().numpy())
                    val_labels.append(val_label)

                    # Split the categories
                    if sum(val_label == 1) > 0:
                        val_loss_spindles_epoch.append(
                            loss_step[val_label == 1].mean().cpu().detach().numpy())

                    if sum(val_label == 0) > 0:
                        val_loss_non_spindles_epoch.append(
                            loss_step[val_label == 0].mean().cpu().detach().numpy())

                    validation_losses_epoch.append(loss_step.mean().item())

            # Compute the F1 score for the spindles
            val_labels = torch.tensor(val_labels)
            val_preds = torch.tensor(val_preds)

            val_labels_flat = val_labels.T.flatten(start_dim=0, end_dim=1)
            val_preds_flat = val_preds.T.flatten(start_dim=0, end_dim=1)

            val_labels_flat = val_labels_flat.cpu().detach().numpy()
            val_preds_flat = val_preds_flat.cpu().detach().numpy()

            metrics = spindle_metrics(
                val_labels_flat,
                val_preds_flat,
                threshold=0.5,
                sampling_rate=6,
                min_label_time=0.5)

            logger.log({
                'val_adap_f1': metrics['f1'],
                'val_adap_precision': metrics['precision'],
                'val_adap_recall': metrics['recall'],
                'val_loss_spindle': np.mean(val_loss_spindles_epoch),
                'val_loss_non_spindle': np.mean(val_loss_non_spindles_epoch),
                'val_loss': np.mean(validation_losses_epoch),
                'used_threshold': used_threshold,
            }, step=index)

    # Run one last time the online spindle detection
    adap_dataset.run_spindle_detection(adap_dataset.last_wamsley_run)

    # Sleep staging metrics:
    ss_preds = adap_dataset.ss_pred_buffer
    ss_labels = adap_dataset.ss_label_buffer
    ss_preds = torch.tensor(ss_preds)
    ss_labels = torch.tensor(ss_labels)
    ss_metrics = staging_metrics(
        ss_labels,
        ss_preds)

    logger.summary['ss_metrics'] = ss_metrics

    # Compute the metrics for the online Lacourse
    true_labels = torch.tensor(adap_dataset.spindle_labels)
    online_lacourse_preds = adap_dataset.get_lacourse_spindle_vector()
    online_lacourse_metrics = spindle_metrics(
        true_labels,
        online_lacourse_preds,
        threshold=0.5,
        sampling_rate=250,
        min_label_time=0.5)

    logger.summary['online_lacourse_metrics'] = online_lacourse_metrics

    # Compute the real_life metrics for spindle detection
    spindle_preds_real = torch.tensor(
        adap_dataset.prediction_buffer).type(torch.LongTensor)
    spindle_labels = torch.tensor(
        adap_dataset.spindle_labels).type(torch.LongTensor)
    spindle_metrics_real = spindle_metrics(
        spindle_labels,
        spindle_preds_real,
        threshold=0.5,
        sampling_rate=250,
        min_label_time=0.5)

    logger.summary['spindle_metrics_real'] = spindle_metrics_real

    all_metrics = {
        # Metrics of the sleep staging
        'ss_metrics': ss_metrics[0],
        'ss_confusion_matrix': ss_metrics[1],
        # Metrics of the spindles using adaptable threshold
        'detect_spindle_metrics': spindle_metrics_real,
        # Metrics of the online wamsley spindle detection
        'online_lacourse_metrics': online_lacourse_metrics,
        # Metrics of the spindles using different thresholds
        # 'multi_threshold_metrics': spindle_mets,
        # All the thresholds used adapted to the data using wamsley
    }

    return all_metrics, net_inference


def average_weights(modelA, modelB, alpha=0.5):
    """
    Average the weights of two models
    """
    new_model = copy.deepcopy(modelA)
    sdA = modelA.state_dict()
    sdB = modelB.state_dict()

    for key in sdA.keys():
        if not torch.all(sdA[key] == sdB[key]):
            sdA[key] = alpha * sdA[key] + (1 - alpha) * sdB[key]

    new_model.load_state_dict(sdA)

    return new_model


def spindle_metrics(labels, preds, ss_labels=None, threshold=0.5, sampling_rate=250, min_label_time=0.5):
    assert len(labels) == len(
        preds), "Labels and predictions are not the same length"

    preds = preds > threshold

    onsets_labels = get_spindle_onsets(
        labels, sampling_rate=sampling_rate, min_label_time=min_label_time)
    onsets_preds = get_spindle_onsets(
        preds, sampling_rate=sampling_rate, min_label_time=min_label_time)
    # Compute the metrics
    precision, recall, f1, tp, fp, fn, closest = binary_f1_score(
        onsets_labels, onsets_preds, sampling_rate=sampling_rate, min_time_positive=min_label_time)
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }

    # Remove all spindles that are in wrong sleep stages
    if ss_labels is not None:
        assert len(labels) == len(
            ss_labels), "Labels and predictions are not the same length"
        onsets_preds_corrected = []
        for spindle in onsets_preds:
            ss_label = ss_labels[spindle]
            if ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3"):
                onsets_preds_corrected.append(spindle)

        precision_corrected, recall_corrected, f1_corrected, tp_corrected, fp_corrected, fn_corrected, closest_corrected = binary_f1_score(
            onsets_labels, onsets_preds_corrected, sampling_rate=sampling_rate, min_time_positive=min_label_time)

        metrics_corrected = {
            'precision_corrected': precision_corrected,
            'recall_corrected': recall_corrected,
            'f1_corrected': f1_corrected,
            'tp_corrected': tp_corrected,
            'fp_corrected': fp_corrected,
            'fn_corrected': fn_corrected,
        }

        metrics = {**metrics, **metrics_corrected}

    return metrics


def binary_search_max(func, low=0.0, high=1.0, tol=0.001):
    max_val = -float('inf')
    max_thresh = low
    num_counts = 0

    while high - low > tol:
        num_counts += 1
        mid = (low + high) / 2
        val = func(mid)

        if val > max_val:
            max_val = val
            max_thresh = mid

        # Evaluate the function at two points around the middle threshold
        val_minus = func(max(mid - 0.1, 0.01))
        val_plus = func(min(mid + 0.1, 0.99))

        # Determine the direction in which the function is increasing
        if val_minus < val_plus:
            low = (low + mid) / 2
        else:
            high = (high + mid) / 2

    print(f"Ran binary search {num_counts} times")

    return max_val, max_thresh


def hierarchical_search(func, low=0.0, high=1.0):

    def grid_search(grid):
        max_val = -float('inf')
        max_thresh = -1
        for point in grid:
            val = func(point)
            if val > max_val:
                max_val = val
                max_thresh = point

        return max_val, max_thresh

    # First, we search for a coarse maximum
    grid_points = np.arange(low, high, 0.1)
    max_val, max_thresh = grid_search(grid_points)

    # Then, we search for a finer maximum
    grid_points = np.arange(max_thresh - 0.1, max_thresh + 0.1, 0.01)
    max_val, max_thresh = grid_search(grid_points)

    # Finally, we search for the finest maximum
    grid_points = np.arange(max_thresh - 0.01, max_thresh + 0.01, 0.001)
    max_val, max_thresh = grid_search(grid_points)

    return max_val, max_thresh


def staging_metrics(labels, preds):
    if len(preds.shape) > 1:
        ss_preds_all = torch.argmax(preds, dim=1)
    else:
        ss_preds_all = preds.cpu().detach()

    # We remove all indexes where the label is 5 (unknown)
    mask = labels != 5
    ss_labels = labels[mask]
    ss_preds = ss_preds_all[mask]

    # Compute the metrics for sleep staging using sklearn classification report
    report_ss = classification_report(
        ss_labels,
        ss_preds,
        output_dict=True,
    )

    # Get the confusion matrix
    cm = confusion_matrix(
        ss_labels,
        ss_preds,
        labels=[0, 1, 2, 3, 4],
    )

    return report_ss, cm, ss_preds_all


def parse_config():
    """
    Parses the config file
    """
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--subject_id', type=str, default='01-01-0001',
                        help='Subject on which to run the experiment')
    parser.add_argument('--model_path', type=str, default='larger_and_hidden_on_loss',
                        help='Model for the starting point of the model')
    parser.add_argument('--dataset_path', type=str, default='/project/MASS/mass_spindles_dataset/',
                        help='Path to the dataset')
    parser.add_argument('--experiment_name', type=str,
                        default='test', help='Name of the model')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Seed for the experiment')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Id of the worker')
    parser.add_argument('--job_id', type=int, default=0,
                        help='Id of the job used for the output file naming scheme')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers used to compute which arguments to run')
    args = parser.parse_args()

    return args


def parse_worker_subject_div(subjects, total_workers, worker_id):
    # Calculate the number of subjects per worker
    subjects_per_worker = len(subjects) // total_workers

    # Calculate the range of subjects for this worker
    start_index = worker_id * subjects_per_worker
    end_index = (worker_id + 1) * subjects_per_worker

    # Ensure the last worker handles any remaining subjects
    if worker_id == total_workers - 1:
        end_index = len(subjects)

    # Extract the subjects for this worker based on the calculated range
    worker_subjects = subjects[start_index:end_index]

    return worker_subjects


def dataloader_from_subject(subject, dataset_path, config, val):
    dataset = MassDataset(
        dataset_path,
        subjects=subject,
        window_size=config['window_size'],
        seq_stride=config['seq_stride'],
        seq_len=1,
        use_filtered=False,
        compute_spindle_labels=False,
        sampleable='spindles',
        wamsley_config=config['wamsley_config']
    )

    if val:
        sampler = MassConsecutiveSampler(
            dataset,
            seq_stride=config['seq_stride'],
            segment_len=1000,
            max_batch_size=512,
            late=False
        )
    else:
        sampler = MassConsecutiveSampler(
            dataset,
            seq_stride=config['seq_stride'],
            segment_len=(len(dataset) // config['seq_stride']) - 1,
            # segment_len=1000,
            max_batch_size=1,
            random=False,
        )

    batch_size = sampler.get_batch_size()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1 if not val else batch_size,
        sampler=sampler,
        num_workers=0,
        shuffle=False)

    return dataloader


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


def get_config(index=0, replay_subjects=None):
    config = {
        'experiment_name': f'config_{index}',
        'num_subjects': 1,
        'train': True if index == 2 or index == 3 else False,
        'seq_len': net.config['seq_len'],
        'seq_stride': net.config['seq_stride'],
        'window_size': net.config['window_size'],
        'lr': 0.00001,
        'adam_w': 0.000,
        'alpha_training': 0.5,  # 1.0 -> Do not use learned, 0.0 -> Keep only learned weights
        'hidden_size': net.config['hidden_size'],
        'nb_rnn_layers': net.config['nb_rnn_layers'],
        # Whether to use the adaptable threshold in the detection of spindles with NN Model
        'adapt_threshold_detect': True if index == 1 or index == 3 else False,
        # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
        'adapt_threshold_wamsley': True,
        # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
        'learn_wamsley': True,
        # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
        'use_ss_label': True if index == 6 else False,
        'use_mask_wamsley': False if index == 4 else True,
        # Smoothing for the sleep staging (WIP)
        'use_ss_smoothing': False,
        'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
        # Thresholds for the adaptable threshold
        'min_threshold': 0.0,
        'max_threshold': 1.0,
        'starting_threshold': 0.5,
        # Interval between each run of online wamsley, threshold adaptation and finetuning if required (in minutes)
        'adaptation_interval': 60,
        # Weights for the sampling of the different spindle in the finetuning in order: [fn, tp, fp, tn]
        'sample_weights': [0.25, 0.25, 0.25, 0.25],
        # Batch size for the finetuning, the bigger the less time it takes to finetune
        'batch_size': 512,
        'num_batches_train': 1000,
        'wamsley_config': {
            'fixed': False,
            'squarred': False,
            'remove_outliers': False,
            'threshold_multiplier': 4.5,
            'sampling_rate': 250,
        },
        'use_replay': False,
        # Select ten random subjects on which the model was trained
        'replay_subjects': np.random.choice(net.config['subjects_train'], 10) if replay_subjects is None else replay_subjects,
        # 'replay_subjects': net.config['subjects_train'],
        'replay_multiplier': 1,
        'freeze_embeddings': False,
        'freeze_classifier': True,
    }

    return config


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed
    seed = 40
    set_seeds(seed)

    group_name = 'adapt_avg'
    run_id = 'both_cc_olddl_lac_newdropout_32142'
    exp_name_val = 'new_avg_freeze_clas'
    # Each worker only does its subjects
    worker_id = args.worker_id
    # run_id_old = "both_cc_smallLR_1706210166"
    unique_id = f"{int(time.time())}"[5:]
    net, run = load_model_mass(
        f"Loading_subjects_{worker_id}_{unique_id}", run_id=run_id, group_name='ModelLoaders')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path
    # loader = SubjectLoader(
    #     os.path.join(dataset_path, 'subject_info.csv'))
    # subjects = loader.select_random_subjects(config['num_subjects'])

    # Taking only the subjects on which the model wasnt trained to avoid data contamination
    subjects = net.config['subjects_test']

    subjects = parse_worker_subject_div(
        subjects, args.num_workers, worker_id)

    all_configs = [get_config(i) for i in [4, 5, 6]]
    # all_configs = [get_config(1, replay_subjects=[subjects[0]])]
    # subjects = [subjects[0]]

    results = {}
    # subjects = ['01-03-0025']

    run.finish()

    # print(f"Doing subjects: {subjects}")
    for subject_id in subjects:
        print(f"Running subject {subject_id}")
        results[subject_id] = {}
        for config in all_configs:
            print(f"Running config {config['experiment_name']}")
            unique_id = f"{int(time.time())}"[5:]
            net, run = load_model_mass(
                f"{exp_name_val}_{config['experiment_name']}_{subject_id}_{unique_id}", run_id=run_id, group_name=group_name)

            if config['freeze_embeddings']:
                net.freeze_embeddings()

            if config['freeze_classifier']:
                net.freeze_classifiers()

            config['subject'] = subject_id
            run.config.update(config)
            dataloader = dataloader_from_subject(
                [subject_id], dataset_path, config, val=False)
            val_dataloader = dataloader_from_subject(
                [subject_id], dataset_path, config, val=True)
            config['subject'] = subject_id
            metrics, net_copy = run_adaptation(
                dataloader,
                val_dataloader,
                net,
                device,
                config,
                config['train'],
                logger=run)

            results[subject_id][config['experiment_name']] = {
                'config': config,
                'metrics': metrics
            }

            # Save the results to json file with indentation
            with open(f'results_{exp_name_val}.json', 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)

            # Save the results to wandb as well
            run.save(f'results_{exp_name_val}.json')
            run.finish()

    # Save the results to json file with indentation
    with open(f'experiment_result_worker_{worker_id}.json', 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
