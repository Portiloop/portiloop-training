import argparse
import random
import time
import torch
from torch import nn
from torch import optim
import copy

from portiloop_software.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, read_pretraining_dataset, read_spindle_trains_labels
from portiloop_software.portiloop_python.ANN.models.lstm import get_trained_model
from portiloop_software.portiloop_python.ANN.utils import get_configs, get_metrics


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
            self.add_sample(self.window_buffer[-self.seq_len:], self.label_buffer[-1])

    def spindle_percentage(self):
        sum_spindles = sum([i[1] for i in self.samples if i[1] == 1])
        return sum_spindles / len(self)

    def has_samples(self):
        return len(self.spindle_indexes) > 0 and len(self.non_spindle_indexes) > 0 


def run_adaptation(dataloader, net, device, config):
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
    optimizer = optim.AdamW(net_copy.parameters(), lr=config['lr_adam'], weight_decay=config['adam_w'])
    criterion = nn.BCELoss(reduction='none')

    loss = 0
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1, config['hidden_size']), device=device)

    # Run through the dataloader
    out_grus = None
    out_loss = 0
    for index, info in enumerate(dataloader):

        with torch.no_grad():
            # if index > 10000:
            #     break

            # print(f"Batch {index}")
            # Get the data and labels
            window_data, _, _, window_labels = info
            # window_data = window_data.unsqueeze(0)
            # window_labels = window_labels.unsqueeze(0)

            adap_dataset.add_window(window_data, window_labels)
            
            window_data = window_data.to(device)
            window_labels = window_labels.to(device)

            if index % 10000 == 0:
                print(f"Doing index: {index}/{len(dataloader.sampler)}")

            # Get the output of the network
            output, h1, _ = net_copy(window_data, h1)

            # Compute the loss
            output = output.squeeze(-1)
            window_labels = window_labels.float()
            
            # Update the loss
            out_loss += criterion(output, window_labels).detach().item()
            n += 1

            # Get the predictions
            output = (output >= 0.5)

            # Concatenate the predictions
            window_labels_total = torch.cat([window_labels_total, window_labels])
            output_total = torch.cat([output_total, output])

        # Training loop for the adaptation
        if index % 100 == 0 and adap_dataset.has_samples():
            train_sample, train_label = next(iter(adap_dataloader))
            train_sample = train_sample.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()

            # Get the output of the network
            h_zero = torch.zeros((config['nb_rnn_layers'], train_sample.size(0), config['hidden_size']), device=device)
            output, _, _ = net_copy(train_sample, h_zero)
            
            # Compute the loss
            output = output.squeeze(-1)
            train_label = train_label.squeeze(-1).float()
            loss = criterion(output, train_label)
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()

    print(adap_dataset.spindle_percentage())

    # Compute metrics
    loss /= n
    acc = (output_total == window_labels_total).float().mean()
    output_total = output_total.float()
    window_labels_total = window_labels_total.float()

    # Get the true positives, true negatives, false positives and false negatives
    tp = (window_labels_total * output_total)
    tn = ((1 - window_labels_total) * (1 - output_total))
    fp = ((1 - window_labels_total) * output_total)
    fn = (window_labels_total * (1 - output_total))

    return output_total, window_labels_total, loss, acc, tp, tn, fp, fn, net_copy


def parse_config():
    """
    Parses the config file
    """
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--subject_id', type=str, default='01-01-0001', help='Subject on which to run the experiment')
    parser.add_argument('--model_path', type=str, default='test_filtered_MASS', help='Model for the starting point of the model')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the model')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for the experiment')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    config = get_configs(args.experiment_name, False, seed)
    config['subject_id'] = args.subject_id

    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    data = read_pretraining_dataset(config['MASS_dir'], patients_to_keep=[args.subject_id])

    assert args.subject_id in data.keys(), 'Subject not in the dataset'
    assert args.subject_id in labels.keys(), 'Subject not in the dataset'

    dataset = SingleSubjectDataset(config['subject_id'], data=data, labels=labels, config=config)   
    sampler = SingleSubjectSampler(len(dataset), config['seq_stride'])
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        sampler=sampler, 
        num_workers=0)

    # Load the model
    net = get_trained_model(config, config['path_models'] / args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run the adaptation
    start = time.time()
    # run_adaptation(dataloader, net, device, config)
    output_total, window_labels_total, loss, acc, tp, tn, fp, fn, net_copy = run_adaptation(dataloader, net, device, config)
    end = time.time()
    print('Time: ', end - start)

    # Get the metrics
    f1, precision, recall = get_metrics(tp, fp, fn)

    # Print the results
    print('Loss: ', loss)
    print('Accuracy: ', acc)
    print('F1: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
