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

from portiloopml.portiloop_python.ANN.wamsley_utils import binary_f1_score, detect_wamsley, get_spindle_onsets


class AdaptationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
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

    def __iter__(self):
        """
        Returns an iterator over the dataset
        """
        while True:
            # Choose between one of the 4 types of samples
            sample_type = random.randint(0, 3)
            # sample_type = 0
            if sample_type == 0:
                # We pick a false negative
                if len(self.dataset.sampleable_missed_spindles) > 0:
                    index = random.choice(
                        self.dataset.sampleable_missed_spindles)
                    self.stats['false_negatives'] += 1
                    # We remove the index from the sampleable list
                    self.dataset.sampleable_missed_spindles.remove(index)
                else:
                    index = -1
            elif sample_type == 1:
                # We pick a true positive
                if len(self.dataset.sampleable_found_spindles) > 0:
                    index = random.choice(
                        self.dataset.sampleable_found_spindles)
                    # We remove the index from the sampleable list
                    self.stats['true_positives'] += 1
                    # We remove the index from the sampleable list
                    self.dataset.sampleable_found_spindles.remove(index)
                else:
                    index = -1
            elif sample_type == 2:
                # We pick a false positive
                if len(self.dataset.sampleable_false_spindles) > 0:
                    index = random.choice(
                        self.dataset.sampleable_false_spindles)
                    self.stats['false_positives'] += 1
                    # We remove the index from the sampleable list
                    self.dataset.sampleable_false_spindles.remove(index)
                else:
                    index = -1
            else:
                # We pick a true negative
                index = random.randint(
                    0, len(self.dataset.wamsley_buffer) - 1)
                self.stats['true_negatives'] += 1

            # Check if the index is far enough from the begginins of the buffer
            if index < self.dataset.min_signal_len:
                continue

            label = 1 if sample_type == 0 or sample_type == 1 else 0

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
        self.wamsley_interval = 250 * 60 * 10
        self.batch_size = batch_size
        self.samples = []
        self.window_buffer = []
        self.label_buffer = []
        self.spindle_indexes = []
        self.non_spindle_indexes = []
        self.pred_threshold = 0.50

        # signal needed before the last window
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        self.window_size = config['window_size']
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + self.window_size

        # Buffers
        self.wamsley_buffer = []
        self.ss_mask_buffer = []
        self.prediction_buffer = []
        self.prediction_buffer_raw = []
        self.spindle_labels = []
        self.last_wamsley_run = 0

        # Sampleable lists
        # Indexes of false negatives that can be sampled
        self.sampleable_missed_spindles = []
        # Indexes of false positives that can be sampled
        self.sampleable_false_spindles = []
        # Indexes of true positives that can be sampled
        self.sampleable_found_spindles = []
        # We assume that the true negatives are just all the rest given that they are majority

        # Used for tests:
        self.used_thresholds = []
        self.candidate_thresholds = np.arange(
            config['min_threshold'], config['max_threshold'], 0.01)
        self.adapt_threshold_detect = config['adapt_threshold_detect']
        self.adapt_threshold_wamsley = config['adapt_threshold_wamsley']
        self.learn_wamsley = config['learn_wamsley']

    def __getitem__(self, index):
        """
        Returns a sample from the dataset
        """
        # Get the index and the sample type
        index, label = index

        # Get data
        index = index + self.past_signal_len
        signal = torch.Tensor(self.wamsley_buffer[index - self.past_signal_len:index + self.window_size]).unfold(
            0, self.window_size, self.seq_stride)

        # Make sure that the last index of the signal is the same as the label
        # assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        label = torch.Tensor([label]).type(torch.LongTensor)
        signal = signal.unsqueeze(1)

        return signal, label

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.wamsley_buffer) - self.min_signal_len

    def add_window(self, window, ss_label, model_pred, spindle_label, detect_threshold, force_wamsley=False):
        """
        Adds a window to the buffers to make it a sample when enough windows have arrived
        """
        # Check if we are in one of the right sleep stages for the Wamsley Mask
        if (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
            ss_label = True
        else:
            ss_label = False

        # Prepare the data to be added to the buffers
        points_to_add = window.squeeze().tolist() if len(
            self.wamsley_buffer) == 0 else window.squeeze().tolist()[-self.seq_stride:]
        ss_mask_to_add = [ss_label] * len(points_to_add)
        preds_raw_to_add = [model_pred.cpu()] * len(points_to_add)
        preds_to_add = [bool(model_pred.cpu() >= detect_threshold)] * \
            len(points_to_add)
        spindle_label_to_add = [spindle_label] * len(points_to_add)

        # Add to the buffers
        self.wamsley_buffer += points_to_add
        self.ss_mask_buffer += ss_mask_to_add
        self.prediction_buffer += preds_to_add
        self.prediction_buffer_raw += preds_raw_to_add
        self.spindle_labels += spindle_label_to_add

        # Update the last wamsley run counter
        self.last_wamsley_run += len(points_to_add)

        assert len(self.ss_mask_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.prediction_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"

        # Check if we have reached the Wamsley interval
        if self.last_wamsley_run >= self.wamsley_interval or force_wamsley:
            # Reset the counter to 0
            self.last_wamsley_run = 0

            # Run Wamsley on the buffer
            usable_buffer = self.wamsley_buffer[-self.wamsley_interval:]
            usable_mask = self.ss_mask_buffer[-self.wamsley_interval:]
            usable_preds = self.prediction_buffer[-self.wamsley_interval:]
            usable_labels = self.spindle_labels[-self.wamsley_interval:]

            # Get the Wamsley spindles using an adaptable threshold if desired
            if self.adapt_threshold_wamsley:
                wamsley_spindles, threshold, used_threshold = detect_wamsley(
                    np.array(usable_buffer), np.array(usable_mask), thresholds=self.wamsley_thresholds)
            else:
                wamsley_spindles, threshold, used_threshold = detect_wamsley(
                    np.array(usable_buffer), np.array(usable_mask))

            if threshold is not None:
                self.wamsley_thresholds.append(threshold)
                self.used_thresholds.append(used_threshold)

            if len(wamsley_spindles) == 0:
                return

            # Get the indexes of the spindles predictions
            spindle_indexes = np.where(np.array(usable_preds) == 1)[
                0]
            events_array = np.array(wamsley_spindles)
            event_ranges = np.concatenate([np.arange(start, stop)
                                           for start, stop in zip(events_array[:, 0], events_array[:, 2])])

            event_ranges_labels = np.where(np.array(usable_labels) == 1)[0]

            # For testing purposes, we want to sometimes learn from the ground truth instead of Wamsley
            if self.learn_wamsley:
                gt_indexes = np.hstack(event_ranges)
            else:
                if len(event_ranges_labels) == 0:
                    return
                gt_indexes = np.hstack(event_ranges_labels)

            # Get the difference in indexes between the buffer we studied and the total buffer
            index_diff = len(self.wamsley_buffer) - self.wamsley_interval
            # index_diff = 0

            # Get the intersection of the two
            true_positives = np.intersect1d(
                spindle_indexes, gt_indexes) + index_diff
            false_positives = np.setdiff1d(
                spindle_indexes, gt_indexes) + index_diff
            false_negatives = np.setdiff1d(
                gt_indexes, spindle_indexes) + index_diff

            wamsley_spindles = [(i[0] + index_diff, i[1] + index_diff,
                                 i[2] + index_diff) for i in wamsley_spindles]

            self.total_spindles += wamsley_spindles
            self.sampleable_missed_spindles += false_negatives.tolist()
            self.sampleable_false_spindles += false_positives.tolist()
            self.sampleable_found_spindles += true_positives.tolist()

            # Get the best threshold for the model
            if self.adapt_threshold_detect:
                best_threshold, thresh_results = self.get_thresholds()
                return best_threshold

    def get_thresholds(self):
        '''
        Returns the best threshold for the model compared to Wamsley threshold
        '''
        # Get the Wamsley spindles
        spindle_preds = torch.tensor(self.prediction_buffer_raw)
        wamsley_spindles_onsets = [i[0] for i in self.total_spindles]
        spindle_labels = torch.zeros_like(spindle_preds)
        spindle_labels[wamsley_spindles_onsets] = 1

        # Find what the best threshold is for the model
        thresh_results = {}
        for threshold in self.candidate_thresholds:
            spindle_mets = spindle_metrics(
                spindle_labels,
                spindle_preds,
                threshold=threshold,
                sampling_rate=250,
                min_label_time=0.5)
            f1 = spindle_mets['f1']
            thresh_results[threshold] = f1

        # best_f1, best_thresh = binary_search_max(
        #     lambda x: spindle_metrics(
        #         spindle_labels,
        #         spindle_preds,
        #         threshold=x,
        #         sampling_rate=250,
        #         min_label_time=0.5)['f1']
        # )

        # print(f"Best threshold is {best_thresh} with f1 score of {best_f1}")

        best_threshold = max(thresh_results, key=thresh_results.get)

        # print(
        #     f"Best threshold (STUPID) is {best_threshold} with f1 score of {thresh_results[best_threshold]}")

        # return best_thresh, best_f1
        return best_threshold, thresh_results

    def spindle_percentage(self):
        sum_spindles = sum([i[1] for i in self.samples if i[1] == 1])
        return sum_spindles / len(self)

    def has_samples(self):
        return len(self.sampleable_false_spindles) > self.batch_size\
            and len(self.sampleable_found_spindles) > self.batch_size \
            and len(self.sampleable_missed_spindles) > self.batch_size


def run_adaptation(dataloader, net, device, config, train):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """
    batch_size = 32
    # Initialize adaptation dataset stuff
    adap_dataset = AdaptationDataset(config, batch_size)
    sampler = AdaptationSampler(adap_dataset)
    adap_dataloader = torch.utils.data.DataLoader(
        adap_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0)

    # Initialize All the necessary variables
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(net_copy.parameters(),
                            lr=config['lr'], weight_decay=config['adam_w'])
    # optimizer = optim.SGD(net_copy.parameters(), lr=0.000003, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    training_losses = []
    inference_loss = []
    n = 0

    # Keep the sleep staging labels and predictions
    ss_labels = []
    ss_preds_argmaxed = []
    ss_preds = []
    spindle_labels = []
    spindle_preds = []
    used_threshold = 0.5
    used_thresholds_detect = [used_threshold]
    spindle_pred_with_thresh = []

    last_n_ss = deque(maxlen=config['n_ss_smoothing'])

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    # Run through the dataloader
    train_iterations = 0

    counter = 0
    for index, batch in enumerate(tqdm(dataloader)):

        net_copy.eval()
        with torch.no_grad():
            counter += 1

            # Get the data and labels
            window_data, labels = batch

            window_data = window_data.to(device)
            ss_label = labels['sleep_stage'].to(device)
            window_labels = labels['spindle_label'].to(device)

            # Get the output of the network
            spindle_output, ss_output, h1, _ = net_copy(window_data, h1)

            ss_labels.append(ss_label.cpu().item())
            ss_preds.append(ss_output.cpu())

            # spindle_output = spindle_output.squeeze(-1)
            spindle_labels.append(window_labels.cpu().item())
            spindle_preds.append(spindle_output.cpu())

            # Compute the loss
            output = spindle_output.squeeze(-1)
            window_labels = window_labels.float()

            # Update the loss
            inf_loss_step = criterion(output, window_labels).mean().item()
            inference_loss.append(inf_loss_step)
            n += 1

            # Add the window to the adaptation dataset given the sleep stage
            output = torch.sigmoid(output)
            if config['use_ss_label']:
                ss_output = ss_label.cpu().item()
            else:
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

            # if index == len(dataloader) - 1:
            #     force_wamsley = True
            # else:
            #     force_wamsley = False
            ss_preds_argmaxed.append(ss_output)
            new_threshold = adap_dataset.add_window(
                window_data,
                ss_output,
                output,
                spindle_label=window_labels.cpu().item(),
                detect_threshold=used_threshold,
                # force_wamsley=force_wamsley
            )

            spindle_pred_with_thresh.append(
                output.cpu().item() > used_threshold)

            if new_threshold is not None and config['adapt_threshold_detect']:
                used_threshold = new_threshold
                used_thresholds_detect.append(new_threshold)

        # Training loop for the adaptation
        if adap_dataset.has_samples() and train:
            net_copy.train()
            train_iterations += 1
            train_sample, train_label = next(iter(adap_dataloader))
            train_sample = train_sample.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()

            # Get the output of the network
            h_zero = torch.zeros((config['nb_rnn_layers'], train_sample.size(
                0), config['hidden_size']), device=device)
            output, _, _, _ = net_copy(train_sample, h_zero)

            # Compute the loss
            output = output.squeeze(-1)
            train_label = train_label.squeeze(-1).float()
            loss_step = criterion(output, train_label)

            loss_step = loss_step.mean()
            loss_step.backward()
            optimizer.step()
            training_losses.append(loss_step.item())

    # COMPUTE METRICS FOR SUBJECT
    inf_loss = sum(inference_loss)
    inf_loss /= n

    # Compute the metrics for sleep staging
    ss_labels = torch.Tensor(ss_labels).type(torch.LongTensor)
    ss_preds = torch.cat(ss_preds, dim=0)
    ss_preds_argmaxed = torch.tensor(ss_preds_argmaxed)
    ss_metrics = staging_metrics(ss_labels, ss_preds_argmaxed)

    # plt.plot(ss_metrics[2])
    # plt.plot(ss_labels)
    # plt.savefig(f'sleep_staging_all.png')

    # # Create a matplotlib figure for the confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=ss_metrics[1],
    #                               display_labels=MassDataset.get_ss_labels()[:-1])
    # disp.plot()
    # plt.savefig(f"sleep_staging_confusion_matrix.png")

    # Compute the metrics for spindles
    spindle_labels = torch.Tensor(spindle_labels).type(torch.LongTensor)
    spindle_preds = torch.cat(spindle_preds, dim=0)
    spindle_preds = torch.sigmoid(spindle_preds)

    ss_preds_4_spindles = torch.argmax(ss_preds, dim=1)
    thresholds = np.arange(config['min_threshold'],
                           config['max_threshold'], 0.01)
    spindle_mets = {}
    for threshold in thresholds:
        spindle_mets[threshold] = spindle_metrics(
            spindle_labels,
            spindle_preds,
            ss_preds_4_spindles,
            threshold=threshold,
            sampling_rate=6,
            min_label_time=0.5)['f1']

    # Plot the f1 score against the threshold
    # f1s = [spindle_mets[threshold]['f1'] for threshold in thresholds]
    # plt.plot(thresholds, f1s)
    # plt.title('F1 score against threshold')
    # plt.xlabel('Threshold')
    # plt.ylabel('F1 score')
    # plt.savefig(f'f1_score_against_threshold.png')
    # plt.clf()
    # spindle_mets = spindle_metrics(
    #     spindle_labels, spindle_preds, ss_preds_4_spindles)

    # best, thresh_res = adap_dataset.get_thresholds()

    # Compute the metrics for the online Wamsley
    true_labels = torch.tensor(adap_dataset.spindle_labels)
    wamsley_preds = torch.zeros_like(true_labels)
    wamsley_spindles_onsets = [i[0] for i in adap_dataset.total_spindles]
    wamsley_preds[wamsley_spindles_onsets] = 1
    online_wamsley_metrics = spindle_metrics(
        true_labels,
        wamsley_preds,
        threshold=0.5,
        sampling_rate=250,
        min_label_time=0.5)

    # Compute the metrics for the adaptable threshold
    spindle_preds_adaptable = torch.tensor(
        spindle_pred_with_thresh).type(torch.LongTensor)
    spindle_metrics_adaptable_thresh = spindle_metrics(
        spindle_labels,
        spindle_preds_adaptable,
        ss_preds_4_spindles,
        threshold=0.5,
        sampling_rate=6,
        min_label_time=0.5)

    all_metrics = {
        # Metrics of the sleep staging
        'ss_metrics': ss_metrics[0],
        'ss_confusion_matrix': ss_metrics[1],
        # Metrics of the spindles using adaptable threshold
        'detect_spindle_metrics': spindle_metrics_adaptable_thresh,
        # Metrics of the online wamsley spindle detection
        'online_wamsley_metrics': online_wamsley_metrics,
        # Metrics of the spindles using different thresholds
        'multi_threshold_metrics': spindle_mets,
        # All the thresholds used adapted to the data using wamsley
        'used_thresholds': used_thresholds_detect,
        # Training loss of the adaptation
        'training_losses': training_losses,
    }

    return all_metrics, net_copy


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
            low = mid
        else:
            high = mid

    print(f"Ran binary search {num_counts} times")

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
    parser.add_argument('--dataset_path', type=str, default='../../data/mass',
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


def dataloader_from_subject(subject, dataset_path):
    dataset = MassDataset(
        dataset_path,
        subjects=subject,
        window_size=54,
        seq_stride=42,
        seq_len=1,
        use_filtered=False)

    sampler = MassConsecutiveSampler(
        dataset,
        seq_stride=42,
        # segment_len=len(dataset) // 42,
        segment_len=1000,
        max_batch_size=1
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
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


def get_all_configs():
    big_config = [
        {
            'experiment_name': 'BASELINE_NOTHING',
            'num_subjects': 1,
            'train': False,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': False,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': False,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': False,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'BASELINE_WAMSLEY',
            'num_subjects': 1,
            'train': False,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': False,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': False,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'BASELINE_WAMSLEY_GT',
            'num_subjects': 1,
            'train': False,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': False,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': True,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'ADAPT_THRESHOLD_DETECTION',
            'num_subjects': 1,
            'train': False,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': True,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': False,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'TRAIN_WITH_THRESH_ADAPT',
            'num_subjects': 1,
            'train': True,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': True,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': False,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'TRAIN_BASELINE_GT',
            'num_subjects': 1,
            'train': True,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': True,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': False,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': False,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        },
        {
            'experiment_name': 'TRAIN_WITH_SS_GT',
            'num_subjects': 1,
            'train': True,
            'seq_len': 50,
            'seq_stride': 42,
            'window_size': 54,
            'lr': 0.00001,
            'adam_w': 0.0,
            'hidden_size': net.config['hidden_size'],
            'nb_rnn_layers': net.config['nb_rnn_layers'],
            # Whether to use the adaptable threshold in the detection of spindles with NN Model
            'adapt_threshold_detect': True,
            # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
            'adapt_threshold_wamsley': True,
            # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
            'learn_wamsley': True,
            # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
            'use_ss_label': True,
            'use_ss_smoothing': False,
            'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
            # Thresholds for the adaptable threshold
            'min_threshold': 0.0,
            'max_threshold': 1.0,
        }
    ]

    return big_config


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed
    set_seeds(seed)

    net, run = load_model_mass("Adaptation")
    net.freeze_embeddings()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config = {
    #     'experiment_name': 'REAL_BASELINE_NOTHING',
    #     'num_subjects': 1,
    #     'train': False,
    #     'seq_len': 50,
    #     'seq_stride': 42,
    #     'window_size': 54,
    #     'lr': 0.00001,
    #     'adam_w': 0.0,
    #     'hidden_size': net.config['hidden_size'],
    #     'nb_rnn_layers': net.config['nb_rnn_layers'],
    #     # Whether to use the adaptable threshold in the detection of spindles with NN Model
    #     'adapt_threshold_detect': False,
    #     # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
    #     'adapt_threshold_wamsley': False,
    #     # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
    #     'learn_wamsley': True,
    #     # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
    #     'use_ss_label': False,
    #     'use_ss_smoothing': False,
    #     'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
    #     # Thresholds for the adaptable threshold
    #     'min_threshold': 0.0,
    #     'max_threshold': 1.0,
    # }

    dataset_path = args.dataset_path
    # loader = SubjectLoader(
    #     os.path.join(dataset_path, 'subject_info.csv'))
    # subjects = loader.select_random_subjects(config['num_subjects'])

    # Taking only the subjects on which the model wasnt trained to avoid data contamination
    subjects = [
        "01-03-0009",
        "01-05-0022",
        "01-03-0030",
        "01-01-0029",
        "01-02-0018",
        "01-03-0012",
        "01-05-0011",
        "01-05-0002",
        "01-01-0004",
        "01-03-0038",
        "01-03-0031",
        "01-03-0010",
        "01-01-0023",
        "01-01-0020",
        "01-05-0019",
        "01-03-0018",
        "01-01-0008",
        "01-01-0031",
        "01-02-0007",
        "01-01-0038"
    ]
    # subjects = subjects[:config['num_subjects']]
    # subjects = ["01-01-0020"]

    # Each worker only does its subjects
    worker_id = args.worker_id

    subjects = parse_worker_subject_div(
        subjects, args.num_workers, worker_id)

    all_configs = get_all_configs()[:2]

    results = {}

    print(f"Doing subjects: {subjects}")
    for subject_id in subjects:
        print(f"Running subject {subject_id}")
        results[subject_id] = {}
        for config in all_configs:
            dataloader = dataloader_from_subject([subject_id], dataset_path)
            metrics, net_copy = run_adaptation(
                dataloader,
                net,
                device,
                config,
                config['train'])

            results[subject_id][config['experiment_name']] = {
                'config': config,
                'metrics': metrics
            }

    # Save the results to json file with indentation
    unique_id = f"{int(time.time())}"[5:]
    with open(f'experiment_result_worker_{worker_id}.json', 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
