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


class AdaptationSampler2(torch.utils.data.Sampler):
    def __init__(self, spindle_indexes, non_spindle_range, past_signal_len, window_size):
        """
        Sample random items from a dataset
        """
        self.spindle_indexes = spindle_indexes
        self.non_spindle_range = non_spindle_range
        self.past_signal_len = past_signal_len
        self.window_size = window_size

    def __iter__(self):
        """
        Returns an iterator over the dataset
        """
        # Choose the same number of spindles and non spindles
        num_spindles = len(self.spindle_indexes)
        non_spindle_indexes = random.sample(
            range(self.non_spindle_range[0], self.non_spindle_range[1]), num_spindles)
        
        # Add the labels to the indexes
        self.spindle_indexes = [(i - self.past_signal_len - self.window_size + 1, 1) for i in self.spindle_indexes]
        non_spindle_indexes = [(i - self.past_signal_len - self.window_size + 1, 0) for i in non_spindle_indexes]
        indexes = self.spindle_indexes + non_spindle_indexes
        random.shuffle(indexes)

        return iter(indexes)
    
    def __len__(self):
        return len(self.spindle_indexes) * 2


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
        self.label_buffer = []
        self.spindle_indexes = []
        self.non_spindle_indexes = []
        self.pred_threshold = 0.50

        self.use_mask_wamsley = config['use_mask_wamsley']

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
        wamsley_config = config['wamsley_config']
        self.wamsley_func = lambda x, y, z: detect_wamsley(
            x,
            y,
            thresholds=z,
            fixed=wamsley_config['fixed'],
            squarred=wamsley_config['squarred'],
            remove_outliers=wamsley_config['remove_outliers'],
            threshold_multiplier=wamsley_config['threshold_multiplier'],
            sampling_rate=wamsley_config['sampling_rate'],
        )

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
        # Keep the categorical loss to see if it improves in all domains or only some of them
        category = label
        if label == 0 or label == 2:
            label = torch.Tensor([0]).type(torch.LongTensor)
        else:
            label = torch.Tensor([1]).type(torch.LongTensor)

        label = torch.Tensor([label]).type(torch.LongTensor)
        signal = signal.unsqueeze(1)

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

        sampleable_indexes = np.array(self.sampleable_spindles)
        train_spindle_indexes = sampleable_indexes[:split_index_pos]
        val_spindle_indexes = sampleable_indexes[split_index_pos:]

        total_indexes_neg = len(self.wamsley_buffer) 
        split_index_neg = int(total_indexes_neg * split)
        train_indexes = (self.min_signal_len, split_index_neg)
        val_indexes = (split_index_neg, total_indexes_neg)

        return train_indexes, val_indexes, train_spindle_indexes, val_spindle_indexes


    def add_window(self, window, ss_label, model_pred, spindle_label, detect_threshold, force_wamsley=False):
        """
        Adds a window to the buffers to make it a sample when enough windows have arrived
        """
        # Check if we are in one of the right sleep stages for the Wamsley Mask
        if (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
            ss_label = True
        else:
            ss_label = False

        if not self.use_mask_wamsley:
            ss_label = True

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
            wamsley_spindles, threshold, used_threshold, spindle_powers, new_thresholds = self.wamsley_func(
                np.array(usable_buffer),
                np.array(usable_mask),
                self.wamsley_thresholds if self.adapt_threshold_wamsley else None
            )

            # Update the list of thresholds
            if new_thresholds is not None:
                self.wamsley_thresholds = new_thresholds

            # Update the used thresholds for testing purposes
            if threshold is not None and spindle_powers is not None:
                self.wamsley_outs += spindle_powers.tolist()
                self.used_thresholds.append(used_threshold)

            # if len(wamsley_spindles) == 0:
            #     return

            # Get the indexes of the spindles predictions
            spindle_indexes = np.where(np.array(usable_preds) == 1)[
                0]

            # For testing purposes, we want to sometimes learn from the ground truth instead of Wamsley
            if self.learn_wamsley:
                events_array = np.array(wamsley_spindles)
                if len(events_array) == 0:
                    return
                event_ranges = np.concatenate([np.arange(start, stop)
                                               for start, stop in zip(events_array[:, 0], events_array[:, 2])])
                gt_indexes = np.hstack(event_ranges)
            else:
                event_ranges_labels = np.where(np.array(usable_labels) == 1)[0]
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
            self.sampleable_spindles += (gt_indexes + index_diff).tolist()

            new_info = len(false_negatives) + \
                len(true_positives) + len(false_positives)

            print(
                f"Got {new_info} new samples with Wamsley threshold {used_threshold}")
            print(
                f"False Negatives: {len(false_negatives)} | False Positives: {len(false_positives)} | True Positives: {len(true_positives)}")

            # Plot the ratios of the spindles
            self.num_true_positive.append(len(true_positives) / new_info)
            self.num_false_positive.append(len(false_positives) / new_info)
            self.num_false_negative.append(len(false_negatives) / new_info)

            # Plot the ratios
            plt.clf()
            plt.plot(self.num_true_positive, label="True Positives")
            plt.plot(self.num_false_positive, label="False Positives")
            plt.plot(self.num_false_negative, label="False Negatives")
            plt.legend()
            # Axis titles:
            plt.title('Ratios of the spindles')
            plt.xlabel('Training Epochs')
            plt.ylabel('Ratio')
            plt.savefig(f'spindle_ratios.png')

            # Get the best threshold for the model
            if self.adapt_threshold_detect:
                best_threshold, thresh_results = self.get_thresholds()
                return (best_threshold, thresh_results), new_info
            else:
                return None, new_info

    def get_thresholds(self):
        '''
        Returns the best threshold for the model compared to Wamsley threshold
        '''
        # Our predictions
        spindle_preds = torch.tensor(self.prediction_buffer_raw)

        # Get the labels to choose the best threshold
        if self.learn_wamsley:
            wamsley_spindles_onsets = [i[0] for i in self.total_spindles]
            spindle_labels = torch.zeros_like(spindle_preds)
            spindle_labels[wamsley_spindles_onsets] = 1
        else:
            spindle_labels = self.spindle_labels

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
        return len(self.sampleable_false_spindles) > self.batch_size\
            or len(self.sampleable_found_spindles) > self.batch_size \
            or len(self.sampleable_missed_spindles) > self.batch_size


def run_adaptation(dataloader, net, device, config, train):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """

    # Run Wamsley to compare to the online Wamsley
    signal = dataloader.dataset.data[config['subject']]['signal']
    ss_labels = dataloader.dataset.data[config['subject']]['ss_label']

    mask = (ss_labels == 1) | (ss_labels == 2)

    batch_size = config['batch_size']
    # Initialize adaptation dataset stuff
    adap_dataset = AdaptationDataset(config, batch_size)
    # sampler = AdaptationSampler(adap_dataset, config['sample_weights'])
    # adap_dataloader = torch.utils.data.DataLoader(
    #     adap_dataset,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=0)

    wamsley_out = adap_dataset.wamsley_func(
        signal,
        mask,
        None)

    # Initialize All the necessary variables
    net_inference = copy.deepcopy(net)
    net_inference = net_inference.to(device)

    # optimizer = optim.SGD(net_copy.parameters(), lr=0.000003, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    training_losses = []
    validation_losses = []
    inference_loss = []
    train_loss_tp = []
    train_loss_fp = []
    train_loss_fn = []
    train_loss_tn = []
    n = 0

    # Keep the sleep staging labels and predictions
    ss_labels = []
    ss_preds_argmaxed = []
    ss_preds = []
    spindle_labels = []
    spindle_preds = []
    used_threshold = config['starting_threshold']
    used_thresholds_detect = [used_threshold]
    spindle_pred_with_thresh = []

    last_n_ss = deque(maxlen=config['n_ss_smoothing'])

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    # Run through the dataloader
    train_iterations = 0

    counter = 0
    train_now = False
    for index, batch in enumerate(tqdm(dataloader)):

        net_inference.eval()
        with torch.no_grad():
            counter += 1

            # Get the data and labels
            window_data, labels = batch

            window_data = window_data.to(device)
            ss_label = labels['sleep_stage'].to(device)
            window_labels = labels['spindle_label'].to(device)

            # Get the output of the network
            spindle_output, ss_output, h1, _ = net_inference(window_data, h1)

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
            out_dataset = adap_dataset.add_window(
                window_data,
                ss_output,
                output,
                spindle_label=window_labels.cpu().item(),
                detect_threshold=used_threshold,
                # force_wamsley=force_wamsley
            )

            spindle_pred_with_thresh.append(
                output.cpu().item() > used_threshold)

            if out_dataset is not None:
                print(f"Got something at index {index}")
                new_threshold, num_new_info = out_dataset
                if new_threshold is not None and config['adapt_threshold_detect']:
                    used_threshold = new_threshold[0]
                    used_thresholds_detect.append(new_threshold)
                if num_new_info is not None:
                    train_now = True

        # Training loop for the adaptation
        if adap_dataset.has_samples() and train and train_now:

            train_now = False

            train_indexes, val_indexes, train_spindle_indexes, val_spindle_indexes = adap_dataset.get_train_val_split(
                split=0.8)
            
            # Initialize the dataloaders
            train_sampler = AdaptationSampler2(
                train_spindle_indexes, 
                train_indexes,
                window_size=adap_dataset.window_size,
                past_signal_len=adap_dataset.past_signal_len)
            
            val_sampler = AdaptationSampler2(
                val_spindle_indexes, 
                val_indexes,
                window_size=adap_dataset.window_size,
                past_signal_len=adap_dataset.past_signal_len)
            
            train_dataloader = torch.utils.data.DataLoader(
                adap_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0)
            val_dataloader = torch.utils.data.DataLoader(
                adap_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=0)
            
            net_copy = copy.deepcopy(net_inference)
            optimizer = optim.Adam(net_copy.parameters(),
                                   lr=config['lr'], weight_decay=config['adam_w'])

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

                # Split the categories
                train_loss_fn.append(
                    loss_step[category == 0].mean().cpu().detach().numpy())
                train_loss_tp.append(
                    loss_step[category == 1].mean().cpu().detach().numpy())
                # train_loss_fp.append(
                #     loss_step[category == 2].mean().cpu().detach().numpy())
                # train_loss_tn.append(
                #     loss_step[category == 3].mean().cpu().detach().numpy())

                loss_step = loss_step.mean()
                loss_step.backward()
                optimizer.step()

                training_losses.append(loss_step.item())

            # Validation loop
            net_copy.eval()
            val_loss_fn = []
            val_loss_tp = []
            val_loss_fp = []
            val_loss_tn = []
            with torch.no_grad():
                for batch, label, category in tqdm(val_dataloader):
                    val_sample = batch.to(device)
                    val_label = label.to(device)

                    # Get the output of the network
                    h_zero = torch.zeros((config['nb_rnn_layers'], val_sample.size(
                        0), config['hidden_size']), device=device)
                    output, _, _, _ = net_copy(val_sample, h_zero)

                    # Compute the loss
                    output = output.squeeze(-1)
                    val_label = val_label.squeeze(-1).float()
                    loss_step = criterion(output, val_label)

                    # Split the categories
                    val_loss_fn.append(
                        loss_step[category == 0].mean().cpu().detach().numpy())
                    val_loss_tp.append(
                        loss_step[category == 1].mean().cpu().detach().numpy())
                    val_loss_fp.append(
                        loss_step[category == 2].mean().cpu().detach().numpy())
                    val_loss_tn.append(
                        loss_step[category == 3].mean().cpu().detach().numpy())

                    validation_losses.append(loss_step.mean().item())

            # Average the weights of the model
            net_inference = average_weights(net_inference, net_copy, alpha=config['alpha_training'])

            # Plot the losses
            plt.clf()
            plt.plot(train_loss_fn,
                     label="Non-Spindle Loss")
            plt.plot(train_loss_tp,
                     label="Spindle Loss")
            plt.plot(train_loss_fp,
                     label="False Positives")
            plt.plot(train_loss_tn,
                     label="True Negatives")
            plt.legend()
            # Axis titles:
            plt.title('Training Losses')
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'training_losses_cat.png')
            # Plot the losses
            plt.clf()
            plt.plot(training_losses)
            # Axis titles:
            plt.title('Training Losses')
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'training_losses.png')

            # Plot the losses  
            plt.clf()
            plt.plot(val_loss_fn,
                        label="Non-Spindle Loss")
            plt.plot(val_loss_tp,
                        label="Spindle Loss")
            plt.plot(val_loss_fp,
                        label="False Positives")
            plt.plot(val_loss_tn,
                        label="True Negatives")
            plt.legend()
            # Axis titles:
            plt.title('Validation Losses')
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'validation_losses_cat.png')
            # Plot the losses
            plt.clf()
            plt.plot(validation_losses)
            # Axis titles:
            plt.title('Validation Losses')
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'validation_losses.png')


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

    # Plot the losses
    plt.clf()
    # plt.plot(inference_loss, alpha=0.5, label="Losses")
    # Do the moving average of the losses
    window = 1000
    inference_loss_smooth = np.convolve(
        inference_loss, np.ones(window) / window, mode='valid')

    plt.plot(inference_loss_smooth, label="Smoothed")

    # Axis titles:
    plt.title('Inference Losses')
    plt.xlabel('window')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'inference_losses.png')

    plt.clf()
    plt.plot(adap_dataset.used_thresholds, label="Used")
    threshd = [i[0] for i in adap_dataset.wamsley_thresholds]
    plt.plot(threshd, label="Thresh")
    # Plot a vertical line at the desired threshold
    plt.axhline(y=wamsley_out[1], color='r', linestyle='--', label="Desired")
    plt.legend()
    plt.savefig(f'used_thresholds.png')

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


def dataloader_from_subject(subject, dataset_path, config):
    dataset = MassDataset(
        dataset_path,
        subjects=subject,
        window_size=54,
        seq_stride=42,
        seq_len=1,
        use_filtered=False,
        compute_spindle_labels=True,
        wamsley_config=config['wamsley_config']
    )

    sampler = MassConsecutiveSampler(
        dataset,
        seq_stride=42,
        segment_len=len(dataset) // 42,
        # segment_len=10000,
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
    seed = 42
    set_seeds(seed)

    net, run = load_model_mass("Adaptation")
    net.freeze_embeddings()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'experiment_name': 'REAL_BASELINE_NOTHING',
        'num_subjects': 1,
        'train': True,
        'seq_len': net.config['seq_len'],
        'seq_stride': net.config['seq_stride'],
        'window_size': net.config['window_size'],
        'lr': 0.00001,
        'adam_w': 0.1,
        'alpha_training': 0.0,
        'hidden_size': net.config['hidden_size'],
        'nb_rnn_layers': net.config['nb_rnn_layers'],
        # Whether to use the adaptable threshold in the detection of spindles with NN Model
        'adapt_threshold_detect': False,
        # Whether to use the adaptable threshold in the detection of spindles with Wamsley online
        'adapt_threshold_wamsley': True,
        # Decides if we finetune from the ground truth (if false) or from our online Wamsley (if True)
        'learn_wamsley': False,
        # Decides if we use the ground truth labels for sleep scoring (for testing purposes)
        'use_ss_label': True,
        'use_mask_wamsley': True,
        # Smoothing for the sleep staging (WIP)
        'use_ss_smoothing': False,
        'n_ss_smoothing': 50,  # 180 * 42 = 7560, which is about 30 seconds of signal
        # Thresholds for the adaptable threshold
        'min_threshold': 0.0,
        'max_threshold': 1.0,
        'starting_threshold': 0.5,
        # Interval between each run of online wamsley, threshold adaptation and finetuning if required (in minutes)
        'adaptation_interval': 180,
        # Weights for the sampling of the different spindle in the finetuning in order: [fn, tp, fp, tn]
        'sample_weights': [0.25, 0.25, 0.25, 0.25],
        # Batch size for the finetuning, the bigger the less time it takes to finetune
        'batch_size': 64,
        'num_batches_train': 1000,
        'wamsley_config': {
            'fixed': False,
            'squarred': True,
            'remove_outliers': False,
            'threshold_multiplier': 4.5,
            'sampling_rate': 250,
        }
    }

    dataset_path = args.dataset_path
    # loader = SubjectLoader(
    #     os.path.join(dataset_path, 'subject_info.csv'))
    # subjects = loader.select_random_subjects(config['num_subjects'])

    # Taking only the subjects on which the model wasnt trained to avoid data contamination
    subjects = net.config['subjects_test']
    subjects = [subjects[0]]

    # Each worker only does its subjects
    worker_id = args.worker_id

    subjects = parse_worker_subject_div(
        subjects, args.num_workers, worker_id)

    # all_configs = get_all_configs()[:2]

    all_configs = [config]

    results = {}

    # print(f"Doing subjects: {subjects}")
    for subject_id in subjects:
        print(f"Running subject {subject_id}")
        results[subject_id] = {}
        for config in all_configs:
            dataloader = dataloader_from_subject(
                [subject_id], dataset_path, config)
            config['subject'] = subject_id
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
    with open(f'experiment_result_worker_{unique_id}.json', 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
