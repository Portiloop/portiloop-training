import argparse
import json
import random
import time
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
        self.batch_size = batch_size
        self.samples = []
        self.window_buffer = []
        self.label_buffer = []
        self.spindle_indexes = []
        self.non_spindle_indexes = []
        self.seq_len = config['seq_len']

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

    def add_window(self, window, label):
        """
        Adds a window to the window buffer to make it a sample when enough windows have arrived
        """
        # TODO: Determine the label of the window here instead of taking it as input

        self.window_buffer.append(window)
        self.label_buffer.append(label)
        # If we have enough windows to create a new sample, we add it to the dataset
        if len(self.window_buffer) > self.seq_len:
            self.add_sample(
                self.window_buffer[-self.seq_len:], self.label_buffer[-1])

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

    loss = 0
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    # Run through the dataloader
    out_loss = 0
    train_iterations = 0

    counter = 0
    for index, info in enumerate(dataloader):
        net_copy.eval()
        with torch.no_grad():
            # if index > 10000:
            #     break
            # if index % 10000 == 0:
            #     print(f"Doing index: {index}/{len(dataloader.sampler)}")

            # if counter % 715 == 0:
            #     h1 = torch.zeros((config['nb_rnn_layers'], 1, config['hidden_size']), device=device)
            counter += 1

            # print(f"Batch {index}")
            # Get the data and labels
            window_data, _, _, window_labels, ss_label = info
            # window_data = window_data.unsqueeze(0)
            # window_labels = window_labels.unsqueeze(0)
            # if not (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
            #     started = False
            #     continue
            # else:
            #     # Reset the model if we just got into an interesting sleep stage
            #     if not started:
            #         h1 = torch.zeros((config['nb_rnn_layers'], 1, config['hidden_size']), device=device)
            #         started = True

            adap_dataset.add_window(window_data, window_labels)

            window_data = window_data.to(device)
            window_labels = window_labels.to(device)

            # Get the output of the network
            output, h1, _ = net_copy(window_data, h1)

            # Compute the loss
            output = output.squeeze(-1)
            window_labels = window_labels.float()

            # Update the loss
            out_loss += criterion(output, window_labels).detach().item()
            n += 1

            # Get the predictions
            output = (output >= 0.95)

            # if output:
            #     rms_score = RMS_score(window_data.squeeze(0).squeeze(0).cpu().numpy()[])

            if skip_ss:
                if (ss_label == SleepStageDataset.get_labels().index("2") or ss_label == SleepStageDataset.get_labels().index("3")):
                    # Concatenate the predictions
                    window_labels_total = torch.cat(
                        [window_labels_total, window_labels])
                    output_total = torch.cat([output_total, output])
                    assert window_labels_total.size() == output_total.size()
            else:
                window_labels_total = torch.cat(
                    [window_labels_total, window_labels])
                output_total = torch.cat([output_total, output])

        # Training loop for the adaptation
        net_copy.train()
        if index % 10 == 0 and adap_dataset.has_samples() and train:
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

    # Compute metrics
    loss /= n
    print(train_iterations)

    return output_total, window_labels_total, loss, net_copy


def RMS_score(candidate, Fs=250, lowcut=11, highcut=16):

    # Filter the signal
    stopbbanAtt = 60  # stopband attenuation of 60 dB.
    width = .5  # This sets the cutoff width in Hertz
    nyq = 0.5 * Fs
    ntaps, _ = kaiserord(stopbbanAtt, width/nyq)
    atten = kaiser_atten(ntaps, width/nyq)
    beta = kaiser_beta(atten)
    a = 1.0
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq,
                  pass_zero=False, window=('kaiser', beta), scale=False)
    filtered_signal = filtfilt(taps, a, candidate)

    # Get the baseline and the detection window for the RMS
    detect_index = len(candidate) // 2
    size_window = 0.5 * Fs
    baseline_idx = -2 * Fs  # Index compared to the detection window
    baseline = filtered_signal[detect_index +
                               baseline_idx:detect_index + baseline_idx + size_window]
    detection = filtered_signal[detect_index:detect_index + size_window]

    # Calculate the RMS
    baseline_rms = torch.sqrt(torch.mean(torch.square(baseline)))
    detection_rms = torch.sqrt(torch.mean(torch.square(detection)))

    score = detection_rms / baseline_rms
    return score


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

    dataset = SingleSubjectDataset(
        config['subject_id'], data=data, labels=labels, config=config, ss_labels=ss_labels)
    sampler = SingleSubjectSampler(len(dataset), config['seq_stride'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0)

    # Run the adaptation
    start = time.time()
    # run_adaptation(dataloader, net, device, config)
    output_total, window_labels_total, loss, net_copy = run_adaptation(
        dataloader, net, device, config, train, skip_ss=skip_ss)
    end = time.time()
    # print('Time: ', end - start)

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

    set_seeds(seed)

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
    random.seed(seed)
    random.shuffle(all_subjects)
    all_subjects = all_subjects[:NUM_SUBJECTS]

    print(f"ALL SUBJECTS: {all_subjects}")
    # Each worker only does its subjects
    worker_id = args.worker_id
    my_subjects_indexes = parse_worker_subject_div(
        all_subjects, args.num_workers, worker_id)
    # Now, you can use worker_subjects in your script for experiments
    for subject in my_subjects_indexes:
        # Perform experiments for the current subject
        print(f"Worker {worker_id} is working on {subject}")

    # all_subjects = ['01-01-0009']

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
            # experiment_results[exp_index][patient_id] = {}
            # loss, acc, f1, precision, recall, dist_preds, dist_labels, _ = run_subject(
            #     net, patient_id, train, labels, ss_labels, skip_ss)
            # # Average all the f1 scores
            # print('Loss: ', loss)
            # print('Accuracy: ', acc)
            # print('F1: ', f1)
            # print('Precision: ', precision)
            # print('Recall: ', recall)

            # experiment_results[exp_index][patient_id]['acc'] = acc.item()
            # experiment_results[exp_index][patient_id]['f1'] = f1
            # experiment_results[exp_index][patient_id]['precision'] = precision
            # experiment_results[exp_index][patient_id]['recall'] = recall
            # experiment_results[exp_index][patient_id]['dist_preds'] = [
            #     i.tolist() for i in dist_preds]
            # experiment_results[exp_index][patient_id]['dist_labels'] = [
            #     i.tolist() for i in dist_labels]
            experiment_results[exp_index][patient_id] = {}
            experiment_results[exp_index][patient_id]['acc'] = 0.99999

    # Save the results to json file with indentation
    with open(f'adap_results_{worker_id}.json', 'w') as f:
        json.dump(experiment_results, f, indent=4)
