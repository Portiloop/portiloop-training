import argparse
import random
import time
import torch
from torch import nn
from torch import optim
import copy
import numpy as np
from portiloop_ml.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, SleepStageDataset, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloop_ml.portiloop_python.ANN.models.lstm import get_trained_model
from portiloop_ml.portiloop_python.ANN.utils import get_configs, get_metrics
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
                yield random.choice(self.dataset.spindle_indexes)
            else:
                yield random.choice(self.dataset.non_spindle_indexes)


class AdaptationDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        """
        Store items from a dataset 
        """
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
        return self.samples[index]

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
        return len(self.spindle_indexes) > 0 and len(self.non_spindle_indexes) > 0


def run_adaptation(dataloader, net, device, config, train, skip=False):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """
    # Initialize adaptation dataset stuff
    adap_dataset = AdaptationDataset(config)
    adap_dataloader = torch.utils.data.DataLoader(
        adap_dataset,
        batch_size=32,
        sampler=AdaptationSampler(adap_dataset),
        num_workers=0)
    # sampler = AdaptationSampler(adap_dataset)

    # Initialize All the necessary variables
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.train()

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(net_copy.parameters(
    ), lr=config['lr_adam'], weight_decay=config['adam_w'])
    criterion = nn.BCELoss(reduction='none')

    loss = 0
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1,
                     config['hidden_size']), device=device)

    # Run through the dataloader
    out_grus = None
    out_loss = 0
    started = False

    counter = 0
    for index, info in enumerate(dataloader):

        with torch.no_grad():
            # if index > 10000:
            #     break
            if index % 10000 == 0:
                print(f"Doing index: {index}/{len(dataloader.sampler)}")

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

            if skip:
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
        if index % 10 == 0 and adap_dataset.has_samples() and train:
            print("training")
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
            loss = criterion(output, train_label)

            loss = loss.mean()
            loss.backward()
            optimizer.step()

    # Compute metrics
    loss /= n

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
    args = parser.parse_args()

    return args


def run_subject(net, subject_id, train, labels, ss_labels):
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
        dataloader, net, device, config, train)
    end = time.time()
    print('Time: ', end - start)

    print("Distribution of the predictions:")
    print(np.unique(output_total.cpu().numpy(), return_counts=True))
    print("Distribution of the labels:")
    print(np.unique(window_labels_total.cpu().numpy(), return_counts=True))

    # Get the metrics
    acc, f1, precision, recall = get_metrics(output_total, window_labels_total)

    return loss, acc, f1, precision, recall, net_copy


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    config = get_configs(args.experiment_name, False, seed)
    # config['nb_conv_layers'] = 4
    # config['hidden_size'] = 64
    # config['nb_rnn_layers'] = 4

    # Load the model
    net = get_trained_model(config, config['path_models'] / args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run some testing on subject 1
    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    ss_labels = read_sleep_staging_labels(config['path_dataset'])
    f1s = []
    # for index, patient_id in enumerate(ss_labels.keys()):
    loss, acc, f1, precision, recall, net = run_subject(
        net, '01-02-0019', False, labels, ss_labels)
    # Average all the f1 scores
    print('Subject ', '01-02-0019')
    print('Loss: ', loss)
    print('Accuracy: ', acc)
    print('F1: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
    f1s.append(f1)

    #     if index > 20:
    #         break

    print('Average F1: ', sum(f1s) / len(f1s))

    # # Print the results
    # print('Subject 1')
    # print('Loss: ', loss)
    # print('Accuracy: ', acc)
    # print('F1: ', f1)
    # print('Precision: ', precision)
    # print('Recall: ', recall)

    # # Run the adaptation on subject 2
    # loss, acc, f1, precision, recall, net = run_subject(net, '01-01-0002', True)

    # # Print the results
    # print('Subject 2')
    # print('Loss: ', loss)
    # print('Accuracy: ', acc)
    # print('F1: ', f1)
    # print('Precision: ', precision)
    # print('Recall: ', recall)

    # # Run some testing on subject 1
    # loss, acc, f1, precision, recall, net = run_subject(net, '01-01-0001', False)
    # print('Subject 1 - Post subject 2')
    # print('Loss: ', loss)
    # print('Accuracy: ', acc)
    # print('F1: ', f1)
    # print('Precision: ', precision)
    # print('Recall: ', recall)
