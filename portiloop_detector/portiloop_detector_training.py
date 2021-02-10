# all imports

from pathlib import Path
from math import floor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import time
from torch.utils.data.sampler import Sampler
from random import randint, seed
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
import pandas as pd
import copy
import wandb
from argparse import ArgumentParser

# all constants (no hyperparameters here!)

filename_dataset = "dataset_big_250_matlab.txt"
path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
FULL_SPINDLE = True
# precision_validation_factor = 0.9
div_val_samp = 50


# all classes and functions:

class SignalDataset(Dataset):
    def __init__(self, filename, path, window_size=64, fe=256, min_length=15, seq_len=5, seq_stride=5, start_ratio=0.0, end_ratio=1.0):
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename
        self.min_length = min_length

        self.data = np.transpose(pd.read_csv(self.path_file, header=None).to_numpy())
        len_data = np.shape(self.data)[1]
        self.data = self.data[:, int(start_ratio * len_data):int(end_ratio * len_data)]
        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.seq_len = seq_len  # 1 means single sample / no sequence ?
        self.idx_stride = seq_stride
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[1][idx + self.window_size - 1] == -1  # that are not ending in an unlabeled zone
                                or self.data[1][idx + self.window_size - self.min_length] == -1  # nor with a min_length starting in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here # TODO: I think this can be tighter

        self.length_dataset = len(self.indices)
        self.labels = torch.tensor([0, 1], dtype=torch.long)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):

        assert 0 <= idx <= self.length_dataset, f"Index out of range ({idx}/{self.length_dataset})."
        idx = self.indices[idx]
        assert self.data[1][idx + self.window_size - 1] != -1 and self.data[1][idx + self.window_size - self.min_length] != -1, f"Bad index: {idx}."

        signal_seq = self.full_signal[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)

        if not FULL_SPINDLE:
            label = 1 if self.data[1][idx + self.window_size - 1] == 1 and self.data[1][idx + self.window_size - self.min_length] != 1 else 0
        else:
            label = 1 if self.data[1][idx + self.window_size - 1] == 1 and self.data[1][idx + self.window_size - self.min_length] == 1 else 0
        label = self.labels[label]

        return signal_seq, label

    def is_spindle(self, idx):
        assert 0 <= idx <= self.length_dataset, f"Index out of range ({idx}/{self.length_dataset})."
        idx = self.indices[idx]
        if not FULL_SPINDLE:
            return True if self.data[1][idx + self.window_size - 1] == 1 and self.data[1][idx + self.window_size - self.min_length] != 1 else False
        else:
            return True if self.data[1][idx + self.window_size - 1] == 1 and self.data[1][idx + self.window_size - self.min_length] == 1 else False


def get_class_idxs(dataset):
    """
    Directly outputs idx_true and idx_false arrays
    """
    length_dataset = len(dataset)

    nb_true = 0
    nb_false = 0

    idx_true = []
    idx_false = []

    for i in range(length_dataset):
        is_spindle = dataset.is_spindle(i)
        if is_spindle:
            nb_true += 1
            idx_true.append(i)
        else:
            nb_false += 1
            idx_false.append(i)

    assert len(dataset) == nb_true + nb_false, f"Bad length dataset"

    return np.array(idx_true), np.array(idx_false)


# Sampler avec liste et sans rand liste

class RandomSampler(Sampler):
    """
    Samples elements randomly and evenly between the two classes.
    The sampling happens WITH replacement.
    __iter__ stops after an arbitrary number of iterations = batch_size_list * nb_batch
    Arguments:
      data_source (Dataset): dataset to sample from
      idx_true: np.array
      idx_false: np.array
      batch_size (int)
      nb_batch (int, optional): number of iteration before end of __iter__(), this defaults to len(data_source)
    """

    def __init__(self, data_source, idx_true, idx_false, batch_size, nb_batch=None):
        self.data_source = data_source
        self.idx_true = idx_true
        self.idx_false = idx_false
        self.nb_true = self.idx_true.size
        self.nb_false = self.idx_false.size
        self.length = nb_batch * batch_size if nb_batch is not None else len(self.data_source)

    def __iter__(self):
        cur_iter = 0
        seed()
        proba = 0.5  # float(0.5052 * precision_validation_factor)
        while cur_iter < self.length:
            cur_iter += 1
            sample_class = np.random.choice([0, 1], p=[1 - proba, proba])
            if sample_class:  # sample true
                idx_file = randint(0, self.nb_true - 1)
                idx_res = self.idx_true[idx_file]
            else:  # sample false
                idx_file = randint(0, self.nb_false - 1)
                idx_res = self.idx_false[idx_file]

            yield idx_res

    def __len__(self):
        return self.length
        # return len(self.data_source)


# Sampler validation

class ValidationSampler(Sampler):
    """
    __iter__ stops after an arbitrary number of iterations = batch_size_list * nb_batch
    """

    def __init__(self, data_source, nb_samples, seq_stride):
        self.length = nb_samples
        self.seq_stride = seq_stride
        self.last_possible = len(data_source) - self.length * self.seq_stride - 1

    def __iter__(self):
        cur_iter = 0
        seed()
        cur_idx = randint(0, self.last_possible)
        while cur_iter < self.length:
            cur_iter += 1
            yield cur_idx
            cur_idx += self.seq_stride

    def __len__(self):
        return self.length
        # return len(self.data_source)


def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


class PortiloopNetwork2(nn.Module):
    def __init__(self, config_dict):
        super(PortiloopNetwork2, self).__init__()

        RNN = config_dict["RNN"]
        stride_pool = config_dict["stride_pool"]
        stride_conv = config_dict["stride_conv"]
        kernel_conv = config_dict["kernel_conv"]
        kernel_pool = config_dict["kernel_pool"]
        nb_channel = config_dict["nb_channel"]
        hidden_size = config_dict["hidden_size"]
        window_size_s = config_dict["window_size_s"]
        dropout_p = config_dict["dropout"]
        dilation_conv = 1  # config_dict["dilation_conv"]
        dilation_max = 1  # config_dict["dilation_max"]
        fe = config_dict["fe"]

        conv_padding = int(kernel_conv // 2)
        max_padding = int(kernel_pool // 2)
        window_size = int(window_size_s * fe)

        self.RNN = RNN
        nb_out = window_size
        self.x1conv1 = nn.Conv1d(in_channels=1,
                                 out_channels=nb_channel,
                                 kernel_size=kernel_conv,
                                 stride=stride_conv,
                                 padding=conv_padding,
                                 dilation=dilation_conv)
        nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)

        self.x1mp1 = nn.MaxPool1d(kernel_size=kernel_pool,
                                  stride=stride_pool,  # note: in the paper they use 1
                                  padding=max_padding,
                                  dilation=dilation_max)
        nb_out = out_dim(nb_out, max_padding, dilation_max, kernel_pool, stride_pool)

        self.x1conv2 = nn.Conv1d(in_channels=nb_channel,
                                 out_channels=nb_channel,
                                 kernel_size=kernel_conv,
                                 stride=stride_conv,
                                 padding=conv_padding,
                                 dilation=dilation_conv)
        nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)

        self.x1mp2 = nn.MaxPool1d(kernel_size=kernel_pool,
                                  stride=stride_pool,  # note: in the paper they use 1
                                  padding=max_padding,
                                  dilation=dilation_max)
        self.x1dropout2 = nn.Dropout(dropout_p)
        nb_out = out_dim(nb_out, max_padding, dilation_max, kernel_pool, stride_pool)

        self.x1conv3 = nn.Conv1d(in_channels=nb_channel,
                                 out_channels=nb_channel,
                                 kernel_size=kernel_conv,
                                 stride=stride_conv,
                                 padding=conv_padding,
                                 dilation=dilation_conv)
        nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)

        self.x1mp3 = nn.MaxPool1d(kernel_size=kernel_pool,
                                  stride=stride_pool,  # note: in the paper they use 1
                                  padding=max_padding,
                                  dilation=dilation_max)
        self.x1dropout3 = nn.Dropout(dropout_p)
        nb_out = out_dim(nb_out, max_padding, dilation_max, kernel_pool, stride_pool)

        self.x1conv4 = nn.Conv1d(in_channels=nb_channel,
                                 out_channels=nb_channel,
                                 kernel_size=kernel_conv,
                                 stride=stride_conv,
                                 padding=conv_padding,
                                 dilation=dilation_conv)
        nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)

        self.x1mp4 = nn.MaxPool1d(kernel_size=kernel_pool,
                                  stride=stride_pool,  # note: in the paper they use 1
                                  padding=max_padding,
                                  dilation=dilation_max)
        self.x1dropout4 = nn.Dropout(dropout_p)
        nb_out = out_dim(nb_out, max_padding, dilation_max, kernel_pool, stride_pool)

        self.x1conv5 = nn.Conv1d(in_channels=nb_channel,
                                 out_channels=nb_channel,
                                 kernel_size=kernel_conv,
                                 stride=stride_conv,
                                 padding=conv_padding,
                                 dilation=dilation_conv)
        nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)

        self.x1mp5 = nn.MaxPool1d(kernel_size=kernel_pool,
                                  stride=stride_pool,  # note: in the paper they use 1
                                  padding=max_padding,
                                  dilation=dilation_max)
        self.x1dropout5 = nn.Dropout(dropout_p)
        nb_out = out_dim(nb_out, max_padding, dilation_max, kernel_pool, stride_pool)

        # # flatten
        output_cnn_size = int(nb_channel * nb_out)
        fc_size = output_cnn_size
        if RNN:
            self.gru = nn.GRU(input_size=output_cnn_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              dropout=0,
                              batch_first=True)
            fc_size = hidden_size
        self.fc = nn.Linear(in_features=fc_size,
                            out_features=2)

    def forward(self, x1, h):
        (batch_size, sequence_len, features) = x1.shape
        x1 = x1.view(-1, 1, features)
        x1 = self.x1conv1(x1)
        x1 = F.relu(x1)  # in the paper they use ELU
        x1 = self.x1mp1(x1)
        x1 = F.relu(self.x1conv2(x1))
        x1 = self.x1mp2(x1)
        x1 = self.x1dropout2(x1)
        x1 = F.relu(self.x1conv3(x1))
        x1 = self.x1mp3(x1)
        x1 = self.x1dropout3(x1)
        x1 = F.relu(self.x1conv4(x1))
        x1 = self.x1mp4(x1)
        x1 = self.x1dropout4(x1)
        x1 = F.relu(self.x1conv5(x1))
        x1 = self.x1mp5(x1)
        x1 = self.x1dropout5(x1)

        x1 = torch.flatten(x1, start_dim=1, end_dim=-1)
        hn = None
        if self.RNN:
            x1 = x1.view(batch_size, sequence_len, -1)
            x1, hn = self.gru(x1, h)
            x1 = x1[:, -1, :]
        x = self.fc(x1)
        x = F.softmax(x, dim=-1)
        return x, hn


class LoggerWandb:
    def __init__(self, experiment_name, config_dict):
        self.losses_train = []
        self.losses_val = []
        self.accuracies_train = []
        self.accuracies_val = []
        self.best_model = None
        self.best_epoch = 0
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project="portiloop", entity="portiloop", id=experiment_name, resume=None, config=config_dict, reinit=True)

    def log(self,
            accuracy_train,
            loss_train,
            accuracy_validation,
            loss_validation,
            f1_validation,
            precision_validation,
            recall_validation,
            best_accuracy_validation,
            best_epoch,
            best_model,
            accuracy_early_stopping,
            best_epoch_early_stopping):
        self.losses_train.append(loss_train)
        self.accuracies_train.append(accuracy_train)
        self.losses_val.append(loss_validation)
        self.accuracies_val.append(accuracy_validation)
        self.best_model = best_model
        self.best_epoch = best_epoch
        wandb.log({
            "accuracy_train": accuracy_train,
            "loss_train": loss_train,
            "accuracy_validation": accuracy_validation,
            "loss_validation": loss_validation,
            "f1_validation": f1_validation,
            "precision_validation": precision_validation,
            "recall_validation": recall_validation,
            "accuracy_early_stopping": accuracy_early_stopping,
        })
        wandb.run.summary["best_accuracy_validation"] = best_accuracy_validation
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_epoch_early_stopping"] = best_epoch_early_stopping

    def __del__(self):
        self.wandb_run.finish()


def get_accuracy_and_loss_pytorch(dataloader, criterion, net, device, hidden_size):
    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    loss = 0
    n = 0
    h = torch.zeros((1, 1, hidden_size), device=device)
    with torch.no_grad():
        for batch_data in dataloader:
            batch_samples, batch_labels = batch_data
            batch_samples = batch_samples.to(device=device).float()
            batch_labels = batch_labels.to(device=device).long()
            output, h = net(batch_samples, h)

            loss_py = criterion(output, batch_labels)
            loss += loss_py.item()
            acc += (output.max(1)[1] == batch_labels).float().mean()

            if output.ndim == 2:
                output = output.argmax(dim=1)

            tp += (batch_labels * output).sum().to(torch.float32)
            tn += ((1 - batch_labels) * (1 - output)).sum().to(torch.float32)
            fp += ((1 - batch_labels) * output).sum().to(torch.float32)
            fn += (batch_labels * (1 - output)).sum().to(torch.float32)

            n += 1
    acc /= n
    loss /= n
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return acc, loss, f1, precision, recall


# run:

def run(config_dict):
    _t_start = time.time()
    print(f"DEBUG: config_dict: {config_dict}")
    experiment_name = config_dict["experiment_name"]
    nb_epoch_max = config_dict["nb_epoch_max"]
    nb_batch_per_epoch = config_dict["nb_batch_per_epoch"]
    nb_epoch_early_stopping_stop = config_dict["nb_epoch_early_stopping_stop"]
    early_stopping_smoothing_factor = config_dict["early_stopping_smoothing_factor"]
    batch_size = config_dict["batch_size"]
    seq_len = config_dict["seq_len"]
    window_size_s = config_dict["window_size_s"]
    fe = config_dict["fe"]
    seq_stride_s = config_dict["seq_stride_s"]
    lr_adam = config_dict["lr_adam"]
    hidden_size = config_dict["hidden_size"]
    device = config_dict["device"]
    max_duration = config_dict["max_duration"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    ds_train = SignalDataset(filename=filename_dataset,
                             path=path_dataset,
                             window_size=window_size,
                             fe=fe,
                             min_length=15,
                             seq_len=seq_len,
                             seq_stride=seq_stride,
                             start_ratio=0.0,
                             end_ratio=0.9)

    ds_validation = SignalDataset(filename=filename_dataset,
                                  path=path_dataset,
                                  window_size=window_size,
                                  fe=fe,
                                  min_length=15,
                                  seq_len=1,
                                  start_ratio=0.90,
                                  end_ratio=1)

    # ds_test = SignalDataset(filename=filename, path_dataset=path_dataset, window_size=window_size, fe=fe, max_length=15, start_ratio=0.95, end_ratio=1, seq_len=1)

    idx_true, idx_false = get_class_idxs(ds_train)

    samp_train = RandomSampler(ds_train,
                               idx_true=idx_true,
                               idx_false=idx_false,
                               batch_size=batch_size,
                               nb_batch=nb_batch_per_epoch)

    samp_validation = ValidationSampler(ds_validation,
                                        nb_samples=int(len(ds_validation) / max(seq_stride, div_val_samp)),
                                        seq_stride=seq_stride)

    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              sampler=samp_train,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    validation_loader = DataLoader(ds_validation,
                                   batch_size=1,
                                   sampler=samp_validation,
                                   num_workers=0,
                                   pin_memory=True,
                                   shuffle=False)

    # test_loader = DataLoader(ds_test, batch_size_list=1, sampler=samp_validation, num_workers=0, pin_memory=True, shuffle=False)

    net = PortiloopNetwork2(config_dict).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_adam)

    losses_pytorch = []
    logger = LoggerWandb(experiment_name, config_dict)

    epoch = 0
    loss_train = 0
    accuracy_train = 0
    loss_validation = 0
    accuracy_validation = 0
    f1_validation = 0
    recall_validation = 0

    best_accuracy = 0
    best_epoch = 0
    best_model = None

    early_stopping_counter = 0
    accuracy_early_stopping = None
    best_accuracy_early_stopping = 0
    best_epoch_early_stopping = 0

    for epoch in range(nb_epoch_max):

        print(f"DEBUG: epoch:{epoch}")

        accuracy_train = 0
        loss_train = 0
        n = 0

        for batch_data in train_loader:
            batch_samples, batch_labels = batch_data
            batch_samples = batch_samples.to(device=device).float()
            batch_labels = batch_labels.to(device=device).long()

            optimizer.zero_grad()
            output, _ = net(batch_samples, torch.zeros((1, batch_size, hidden_size), device=device))
            loss = criterion(output, batch_labels)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
            accuracy_train += (output.max(1)[1] == batch_labels).float().mean()
            n += 1

        accuracy_train /= n
        loss_train /= n

        accuracy_validation, loss_validation, f1_validation, precision_validation, recall_validation = get_accuracy_and_loss_pytorch(validation_loader, criterion, net, device, hidden_size)

        if accuracy_validation > best_accuracy:
            best_accuracy = accuracy_validation
            best_model = copy.deepcopy(net)
            best_epoch = epoch

        accuracy_early_stopping = 0.0 if accuracy_early_stopping is None else accuracy_validation * early_stopping_smoothing_factor + accuracy_early_stopping * (
                    1.0 - early_stopping_smoothing_factor)

        if accuracy_early_stopping > best_accuracy_early_stopping:
            best_accuracy_early_stopping = accuracy_early_stopping
            early_stopping_counter = 0
            best_epoch_early_stopping = epoch
        else:
            early_stopping_counter += 1

        logger.log(accuracy_train=accuracy_train,
                   loss_train=loss_train,
                   accuracy_validation=accuracy_validation,
                   loss_validation=loss_validation,
                   f1_validation=f1_validation,
                   precision_validation=precision_validation,
                   recall_validation=recall_validation,
                   best_accuracy_validation=best_accuracy,
                   best_epoch=best_epoch,
                   best_model=best_model,
                   accuracy_early_stopping=accuracy_early_stopping,
                   best_epoch_early_stopping=best_epoch_early_stopping)

        if early_stopping_counter > nb_epoch_early_stopping_stop or time.time() - _t_start > max_duration:
            print("Early stopping.")
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    args = parser.parse_args()

    exp_name = args.experiment_name

    # hyperparameters

    batch_size_list = [32, 64, 128, 256, 512, 1024]
    seq_len_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    RNN_list = [True, False]
    RNN_weights = [0.8, 0.2]
    kernel_conv_list = [3, 5, 7, 9]
    kernel_pool_list = [3, 5, 7, 9]
    stride_conv_list = [1, 2, 3]
    stride_pool_list = [1, 2, 3]
    stride_weights = [0.6, 0.2, 0.2]
    nb_channel_list = [5, 10, 20, 30, 40, 50]
    hidden_size_list = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    # dilation = [1, 2, 3]
    dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    windows_size_s_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    seq_stride_s_list = [0.025, 0.05, 0.075, 0.1, 0.125]
    lr_adam_list = [0.005, 0.001, 0.0005, 0.0001, 0.00005]

    config_dict = dict(experiment_name=exp_name,
                       device="cuda:0",
                       nb_epoch_max=1000000,
                       max_duration=int(11.5 * 3600),
                       nb_epoch_early_stopping_stop=100,
                       early_stopping_smoothing_factor=0.2,
                       fe=250,
                       nb_batch_per_epoch=1000)

    config_dict["batch_size"] = np.random.choice(batch_size_list).item()
    config_dict["RNN"] = np.random.choice(RNN_list, p=RNN_weights).item()
    config_dict["seq_len"] = np.random.choice(seq_len_list).item() if config_dict["RNN"] else 1
    config_dict["stride_pool"] = np.random.choice(stride_pool_list, p=stride_weights).item()
    config_dict["stride_conv"] = np.random.choice(stride_conv_list, p=stride_weights).item()
    config_dict["kernel_conv"] = np.random.choice(kernel_conv_list).item()
    config_dict["kernel_pool"] = np.random.choice(kernel_pool_list).item()
    config_dict["nb_channel"] = np.random.choice(nb_channel_list).item()
    config_dict["dropout"] = np.random.choice(dropout_list).item()
    config_dict["hidden_size"] = np.random.choice(hidden_size_list).item()
    config_dict["window_size_s"] = np.random.choice(windows_size_s_list).item()
    config_dict["seq_stride_s"] = np.random.choice(seq_stride_s_list).item()
    config_dict["lr_adam"] = np.random.choice(lr_adam_list).item()

    run(config_dict=config_dict)
