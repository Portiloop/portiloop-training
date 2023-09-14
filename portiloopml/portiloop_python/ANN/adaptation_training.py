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
from portiloopml.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, SleepStageDataset, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloopml.portiloop_python.ANN.lightning_tests import load_model
from portiloopml.portiloop_python.ANN.models.lstm import get_trained_model
from portiloopml.portiloop_python.ANN.utils import get_configs, get_metrics, set_seeds
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, kaiserord, filtfilt

from portiloopml.portiloop_python.ANN.wamsley_utils import _detect_start_end, _merge_close, binary_f1_score, detect_wamsley, morlet_transform, remove_straddlers, smooth, within_duration


class AdaptationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        """
        Sample random items from a dataset
        """
        self.dataset = dataset

    def __iter__(self):
        """
        Returns an iterator over the dataset
        """
        while True:
            toss = random.random()
            if toss > 0.5:
                # Get a random index from the spindle indexes
                choice = random.choice(self.dataset.spindle_indexes)
                # remove the index from the spindle indexes
                self.dataset.spindle_indexes.remove(choice)
                yield choice
            else:
                choice = random.choice(self.dataset.non_spindle_indexes)
                self.dataset.non_spindle_indexes.remove(choice)
                yield choice


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
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        self.wamsley_buffer = []
        self.ss_mask_buffer = []

        # Used for tests:
        self.used_thresholds = []
        self.total_sequence = []
        self.last_wamsley_run = 0

    def __getitem__(self, index):
        """
        Returns a sample from the dataset
        """
        sample_to_return = self.samples[index]
        return sample_to_return

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.samples)

    def add_sample(self, sample, label):
        """
        Takes a list of windows and adds them to the dataset
        """
        sample = torch.stack(sample)
        sample = sample.reshape(sample.size(0), sample.size(-1))
        sample = sample.unsqueeze(1)
        self.samples.append((sample, label))
        if label == 1:
            self.spindle_indexes.append(len(self.samples) - 1)
        else:
            self.non_spindle_indexes.append(len(self.samples) - 1)

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
        self.wamsley_buffer += points_to_add
        self.ss_mask_buffer += ss_mask_to_add

        # Update the last wamsley run counter
        self.last_wamsley_run += len(points_to_add)

        assert len(self.ss_mask_buffer) == len(
            self.wamsley_buffer), "Buffers are not the same length"

        # Check if we have reached the Wamsley interval
        if self.last_wamsley_run >= self.wamsley_interval:
            # Reset the counter to 0
            self.last_wamsley_run = 0

            # Run Wamsley on the buffer
            usable_buffer = self.wamsley_buffer[-self.wamsley_interval:]
            usable_mask = self.ss_mask_buffer[-self.wamsley_interval:]

            wamsley_spindles, threshold, used_threshold = detect_wamsley(
                np.array(usable_buffer), np.array(usable_mask), thresholds=self.wamsley_thresholds)

            if threshold is not None:
                self.wamsley_thresholds.append(threshold)
                self.used_thresholds.append(used_threshold)

            index_diff = len(self.wamsley_buffer) - self.wamsley_interval
            wamsley_spindles = [(i[0] + index_diff, i[1] + index_diff,
                                 i[2] + index_diff) for i in wamsley_spindles]

            self.total_spindles += wamsley_spindles
            # TODO: Add the spindle to the ampleable dataset
            # We want to learn in priority from things we got wrong (aka false negatives and false positives)
            # Need to find a smart way to sample those intelligently

    def spindle_percentage(self):
        sum_spindles = sum([i[1] for i in self.samples if i[1] == 1])
        return sum_spindles / len(self)

    def has_samples(self):
        return len(self.spindle_indexes) > self.batch_size and len(self.non_spindle_indexes) > self.batch_size


def run_adaptation(dataloader, net, device, config, train, skip_ss=False):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """
    batch_size = 4
    # Initialize adaptation dataset stuff
    adap_dataset = AdaptationDataset(config, batch_size)
    adap_dataloader = torch.utils.data.DataLoader(
        adap_dataset,
        batch_size=batch_size,
        sampler=AdaptationSampler(adap_dataset),
        num_workers=0)
    # sampler = AdaptationSampler(adap_dataset)

    # Initialize All the necessary variables
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(net_copy.parameters(
    ), lr=0.00005, weight_decay=config['adam_w'])
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
    for index, info in enumerate(dataloader):
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
            output = (output >= 0.82)

            window_labels_total = torch.cat(
                [window_labels_total, window_labels])
            output_total = torch.cat([output_total, output])

        # Training loop for the adaptation
        if index % 10 == 0 and adap_dataset.has_samples() and train:
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
    # inf_loss = sum(inference_loss)
    # inf_loss /= n
    inf_loss = 0.0

    print(len(adap_dataset.wamsley_buffer))
    print(len(dataloader.dataset.full_signal))

    baseline_signal = dataloader.dataset.full_signal
    buffer_signal = adap_dataset.wamsley_buffer

    return output_total, window_labels_total, inf_loss, net_copy, inference_loss, training_losses, adap_dataset.total_spindles


def parse_config():
    """
    Parses the config file
    """
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--subject_id', type=str, default='01-01-0001',
                        help='Subject on which to run the experiment')
    parser.add_argument('--model_path', type=str, default='no_att_baseline',
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
    output_total, window_labels_total, loss, net_copy, inf_losses, train_losses, online_spindles = run_adaptation(
        dataloader, net, device, config, train, skip_ss=skip_ss)

    # Compare the online spindles to the ground truth
    gt_spindles_onsets = [spindle[0] for spindle in all_wamsley_spindles]
    online_spindles_onsets = [spindle[0] for spindle in online_spindles]
    precision, recall, f1 = binary_f1_score(
        gt_spindles_onsets, online_spindles_onsets, threshold=125)

    # Get the distribution of the predictions and the labels
    dist_preds = np.unique(output_total.cpu().numpy(), return_counts=True)
    dist_labels = np.unique(
        window_labels_total.cpu().numpy(), return_counts=True)
    print("Distribution of the predictions:")
    print(dist_preds)
    print("Distribution of the labels:")
    print(dist_labels)

    # Get the metrics
    acc, f1, precision, recall = get_metrics(output_total, window_labels_total)

    return loss, acc, f1, precision, recall, dist_preds, dist_labels, net_copy


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    # set_seeds(seed)

    config = get_configs(args.experiment_name, False, seed)
    # config['nb_conv_layers'] = 4
    # config['hidden_size'] = 64
    # config['nb_rnn_layers'] = 4

    # Load the model
    net = get_trained_model(config, config['path_models'] / args.model_path)
    # ss_net = load_model('Original_params_1692894764.8012033')
    # ss_net.cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run some testing on subject 1
    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    ss_labels = read_sleep_staging_labels(config['path_dataset'])

    # run_ss(ss_net, '01-02-0019', ss_labels)

    # Experiment parameters:
    params = [
        # (Train, Skip SS)
        # (False, True),
        (False, False),
        # (True, False),
        # (True, True)
    ]

    # Number of subjects to run each experiment over
    NUM_SUBJECTS = 1

    experiment_results = {}
    all_subjects = [subject for subject in labels.keys()
                    if subject in ss_labels.keys()]

    for subject in all_subjects:
        assert subject in labels.keys(), 'Subject not in the dataset'
        assert subject in ss_labels.keys(), 'Subject not in the dataset'

    # Randomly select the subjects
    # random.seed(42)
    # random.shuffle(all_subjects)
    # all_subjects = all_subjects[:NUM_SUBJECTS]

    print(f"ALL SUBJECTS: {all_subjects}")
    # Each worker only does its subjects
    worker_id = args.worker_id
    # my_subjects_indexes = parse_worker_subject_div(
    #     all_subjects, args.num_workers, worker_id)
    my_subjects_indexes = ["01-01-0022"]
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
            loss, acc, f1, precision, recall, dist_preds, dist_labels, _ = run_subject(
                net, patient_id, train, labels, ss_labels, skip_ss)
            # Average all the f1 scores
            print('Loss: ', loss)
            print('Accuracy: ', acc)
            print('F1: ', f1)
            print('Precision: ', precision)
            print('Recall: ', recall)

            experiment_results[exp_index][patient_id]['acc'] = acc.item()
            experiment_results[exp_index][patient_id]['f1'] = f1
            experiment_results[exp_index][patient_id]['precision'] = precision
            experiment_results[exp_index][patient_id]['recall'] = recall
            experiment_results[exp_index][patient_id]['dist_preds'] = [
                i.tolist() for i in dist_preds]
            experiment_results[exp_index][patient_id]['dist_labels'] = [
                i.tolist() for i in dist_labels]

    # Save the results to json file with indentation
    with open(f'adap_results_{args.job_id}_{worker_id}.json', 'w') as f:
        json.dump(experiment_results, f, indent=4)
