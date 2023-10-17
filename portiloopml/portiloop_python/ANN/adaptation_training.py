import argparse
from collections import deque
import json
import random
import time
from matplotlib import pyplot as plt
from scipy import signal
import torch
from torch import nn
from torch import optim
import copy
import numpy as np
from tqdm import tqdm
from portiloopml.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, SleepStageDataset, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloopml.portiloop_python.ANN.lightning_tests import load_model
from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork, get_trained_model
from portiloopml.portiloop_python.ANN.utils import get_configs, get_metrics, set_seeds
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, kaiserord, filtfilt

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
        self.pred_threshold = 0.82

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

    def add_window(self, window, ss_label, model_pred):
        """
        Adds a window to the buffers to make it a sample when enough windows have arrived
        """
        # Check if we are in one of the right sleep stages for the Wamsley Mask
        if (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
            ss_label = True
        else:
            ss_label = False

        # Add to the buffers
        points_to_add = window.squeeze().tolist() if len(
            self.wamsley_buffer) == 0 else window.squeeze().tolist()[-self.seq_stride:]
        ss_mask_to_add = [ss_label] * len(points_to_add)
        preds_to_add = [bool(model_pred.cpu() >= self.pred_threshold)] * \
            len(points_to_add)
        self.wamsley_buffer += points_to_add
        self.ss_mask_buffer += ss_mask_to_add
        self.prediction_buffer += preds_to_add

        # Update the last wamsley run counter
        self.last_wamsley_run += len(points_to_add)

        assert len(self.ss_mask_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"
        assert len(self.prediction_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"

        # Check if we have reached the Wamsley interval
        if self.last_wamsley_run >= self.wamsley_interval:
            # Reset the counter to 0
            self.last_wamsley_run = 0

            # Run Wamsley on the buffer
            usable_buffer = self.wamsley_buffer[-self.wamsley_interval:]
            usable_mask = self.ss_mask_buffer[-self.wamsley_interval:]
            usable_preds = self.prediction_buffer[-self.wamsley_interval:]

            wamsley_spindles, threshold, used_threshold = detect_wamsley(
                np.array(usable_buffer), np.array(usable_mask), thresholds=self.wamsley_thresholds)

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
            gt_indexes = np.hstack(event_ranges)

            # Get the difference in indexes between the buffer we studied and the total buffer
            index_diff = len(self.wamsley_buffer) - self.wamsley_interval

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

    def spindle_percentage(self):
        sum_spindles = sum([i[1] for i in self.samples if i[1] == 1])
        return sum_spindles / len(self)

    def has_samples(self):
        return len(self.sampleable_false_spindles) > self.batch_size\
            and len(self.sampleable_found_spindles) > self.batch_size \
            and len(self.sampleable_missed_spindles) > self.batch_size


def run_adaptation(dataloader, net, device, config, train, skip_ss=False):
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
                            lr=0.000005, weight_decay=config['adam_w'])
    # optimizer = optim.SGD(net_copy.parameters(), lr=0.000003, momentum=0.9)
    criterion = nn.BCELoss(reduction='none')

    training_losses = []
    inference_loss = []
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    # Run through the dataloader
    train_iterations = 0

    counter = 0
    for index, info in enumerate(tqdm(dataloader)):

        net_copy.eval()
        with torch.no_grad():
            counter += 1

            # Get the data and labels
            window_data, _, _, window_labels, ss_label = info

            window_data = window_data.to(device)
            window_labels = window_labels.to(device)

            # Get the output of the network
            output, h1, _ = net_copy(window_data, h1)

            # Compute the loss
            output = output.squeeze(-1)
            window_labels = window_labels.float()

            # Update the loss
            inf_loss_step = criterion(output, window_labels).mean().item()
            inference_loss.append(inf_loss_step)
            n += 1

            # Add the window to the adaptation dataset given the sleep stage
            # This should eventually be replaced with a real sleep staging model
            adap_dataset.add_window(window_data, ss_label, output)

            # Get the predictions
            output = (output >= 0.70)

            window_labels_total = torch.cat(
                [window_labels_total, window_labels])
            output_total = torch.cat([output_total, output])

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
            output, _, _ = net_copy(train_sample, h_zero)

            # Compute the loss
            output = output.squeeze(-1)
            train_label = train_label.squeeze(-1).float()
            loss_step = criterion(output, train_label)

            loss_step = loss_step.mean()
            loss_step.backward()
            optimizer.step()
            training_losses.append(loss_step.item())

    # Compute metrics
    inf_loss = sum(inference_loss)
    inf_loss /= n

    return output_total, window_labels_total, inf_loss, net_copy, inference_loss, training_losses, adap_dataset


def parse_config():
    """
    Parses the config file
    """
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--subject_id', type=str, default='01-01-0001',
                        help='Subject on which to run the experiment')
    parser.add_argument('--model_path', type=str, default='larger_and_hidden_on_loss',
                        help='Model for the starting point of the model')
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


def run_subject(net, subject_id, train, labels, ss_labels, skip_ss=False):
    config['subject_id'] = subject_id

    data = read_pretraining_dataset(
        config['MASS_dir'], patients_to_keep=[subject_id])

    assert subject_id in data.keys(), 'Subject not in the dataset'
    assert subject_id in labels.keys(), 'Subject not in the dataset'

    pre_dataset = SingleSubjectDataset(
        config['subject_id'], data=data, labels=labels, config=config, ss_labels=ss_labels, delete=False)

    mask = (np.array(pre_dataset.full_ss_labels) == SleepStageDataset.get_labels().index(
        "3")) | (np.array(pre_dataset.full_ss_labels) == SleepStageDataset.get_labels().index("2"))

    all_wamsley_spindles, all_threshold, _merge_close = detect_wamsley(
        pre_dataset.full_signal, mask)

    spindle_info = {}
    spindle_info[subject_id] = {
        'onsets': [],
        'offsets': [],
        'labels_num': []
    }
    for spindle in all_wamsley_spindles:
        spindle_info[subject_id]['onsets'].append(spindle[0])
        spindle_info[subject_id]['offsets'].append(spindle[2])
        spindle_info[subject_id]['labels_num'].append(1)

    dataset = SingleSubjectDataset(
        config['subject_id'], data=data, labels=spindle_info, config=config, ss_labels=ss_labels)

    sampler = SingleSubjectSampler(len(dataset), config['seq_stride'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0)

    # Run the adaptation
    start = time.time()

    # run_adaptation(dataloader, net, device, config)
    output_total, window_labels_total, loss, net_copy, inf_losses, train_losses, adap_dataset = run_adaptation(
        dataloader, net, device, config, train, skip_ss=skip_ss)

    end = time.time()
    print(f"Time to run the adaptation: {end - start}")

    # Compare the online spindles to the ground truth
    gt_spindles_onsets = [spindle[0] for spindle in all_wamsley_spindles]
    online_spindles_onsets = get_spindle_onsets(
        adap_dataset.prediction_buffer)
    precision, recall, f1, tp, fp, fn, closest = binary_f1_score(
        gt_spindles_onsets, online_spindles_onsets, threshold=125)

    print(
        f"Found {len(online_spindles_onsets)} spindles, expected {len(gt_spindles_onsets)} spindles")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    if closest != []:
        print(f"Average Distance: {sum(closest) / len(closest)}")

    # Show the loss graph
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.savefig(f'adaptation_training_loss_{subject_id}_{train}.png')
    plt.clf()

    # Compute the moving average of the inference loss
    inf_losses = np.array(inf_losses)
    inf_losses = np.convolve(inf_losses, np.ones(
        (100,)) / 100, mode='valid')
    plt.plot(inf_losses)
    plt.title('Inference Loss')
    plt.xlabel('Inference step')
    plt.ylabel('Loss')
    plt.savefig(f'adaptation_inference_loss_{subject_id}_{train}.png')
    plt.clf()

    # Save all metrics to a dictionary
    metrics = {
        'loss': loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'closest': closest,
        'gt_spindles_onsets': gt_spindles_onsets,
        'online_spindles_onsets': online_spindles_onsets,
        'train_losses': train_losses,
        'inf_losses': inf_losses,
        'used_thresholds': adap_dataset.used_thresholds,
    }

    return metrics


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed
    # seed = 42
    set_seeds(seed)

    config = get_configs(args.experiment_name, False, seed)
    # config['nb_conv_layers'] = 4
    config['hidden_size'] = 64
    config['nb_rnn_layers'] = 3
    config['after_rnn'] = 'hidden'

    # Load the model

    net = get_trained_model(config, config['path_models'] / args.model_path)

    for name, param in net.named_parameters():
        if name not in ['fc.weight', 'fc.bias', 'hidden_fc.weight', 'hidden_fc.bias']:
            param.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run some testing on subject 1
    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    ss_labels = read_sleep_staging_labels(config['path_dataset'])

    # Experiment parameters:
    params = [
        # (Train, Skip SS)
        # (False, True),
        (False, False),
        (True, False),
        # (True, True)
    ]

    # Number of subjects to run each experiment over
    NUM_SUBJECTS = 40

    experiment_results = {}
    all_subjects = [subject for subject in labels.keys()
                    if subject in ss_labels.keys()]

    for subject in all_subjects:
        assert subject in labels.keys(), 'Subject not in the dataset'
        assert subject in ss_labels.keys(), 'Subject not in the dataset'

    # Randomly select the subjects
    random.seed(42)
    random.shuffle(all_subjects)
    all_subjects = all_subjects[:NUM_SUBJECTS]

    # Each worker only does its subjects
    worker_id = args.worker_id
    my_subjects_indexes = parse_worker_subject_div(
        all_subjects, args.num_workers, worker_id)
    # my_subjects_indexes = ["01-05-0015"]
    # Now, you can use worker_subjects in your script for experiments
    for subject in my_subjects_indexes:
        # Perform experiments for the current subject
        print(f"Worker {worker_id} is working on {subject}")

    for exp_index, param in enumerate(params):
        train, skip_ss = param
        print('Experiment ', exp_index)
        print('Train: ', train)
        print('Skip SS: ', skip_ss)

        experiment_results[exp_index] = {}
        experiment_results[exp_index]['train'] = train
        experiment_results[exp_index]['skip_ss'] = skip_ss

        for index, patient_id in enumerate(my_subjects_indexes):
            print('Subject ', patient_id)
            experiment_results[exp_index][patient_id] = {}
            metrics = run_subject(
                net, patient_id, train, labels, ss_labels, skip_ss)

            experiment_results[exp_index][patient_id] = metrics

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # Save the results to json file with indentation
    with open(f'adap_results_{args.job_id}-{worker_id}.json', 'w') as f:
        json.dump(experiment_results, f, indent=4, cls=NpEncoder)
