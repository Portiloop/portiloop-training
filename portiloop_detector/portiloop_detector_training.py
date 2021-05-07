# all imports

from pathlib import Path
from math import floor

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
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

THRESHOLD = 0.2
WANDB_PROJECT = "portiloop-multiple_input"
CLASSIFICATION = False

filename_dataset = "dataset_p1_big_250_matlab_standardized_envelope_pf.txt"
path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
recall_validation_factor = 0.5
precision_validation_factor = 0.5

div_val_samp = 25

# hyperparameters

batch_size_list = [256, 256]
seq_len_list = [40, 40]
kernel_conv_list = [5, 5]
kernel_pool_list = [3, 3]
stride_conv_list = [1, 1]
stride_pool_list = [1, 1]
dilation_conv_list = [1, 1]
dilation_pool_list = [1, 1]
nb_channel_list = [20, 20]
hidden_size_list = [15, 15]
dropout_list = [0.5, 0.5]
windows_size_s_list = [0.25, 0.25]
seq_stride_s_list = [0.1, 0.1]
lr_adam_list = [0.0003, 0.0003]
nb_conv_layers_list = [7, 7]
nb_rnn_layers_list = [1, 1]
first_layer_dropout_list = [False, False]
power_features_input_list = [False, False]
adam_w_list = [0.01, 0.01]
distribution_mode_list = [1, 1]


# all classes and functions:

class SignalDataset(Dataset):
    def __init__(self, filename, path, window_size=64, fe=250, seq_len=5, seq_stride=5, list_subject=None):
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename

        self.data = pd.read_csv(self.path_file, header=None).to_numpy()
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / ((115 + 30) * fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))
        print(f"DEBUG: data shape = {self.data.shape}")
        # if "portiloop" in filename:
        #     split_data = np.array(np.split(self.data, int(len(self.data) / (900 * fe))))  # 900 = nb seconds per sequence in the dataset
        # else:
        #     split_data = np.array(np.split(self.data, int(len(self.data) / ((115 + 30) * fe))))  # 115+30 = nb seconds per sequence in the dataset
        # np.random.seed(0)  # fixed seed value
        # np.random.shuffle(split_data)
        # self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))
        # len_data = np.shape(self.data)[1]
        # self.data = self.data[:, int(start_ratio * len_data):int(end_ratio * len_data)]
        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_envelope = torch.tensor(self.data[1], dtype=torch.float)
        self.seq_len = seq_len  # 1 means single sample / no sequence ?
        self.idx_stride = seq_stride
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size - 1] >= 0, f"Bad index: {idx}."

        signal_seq = self.full_signal[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)
        envelope_seq = self.full_envelope[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)

        ratio_pf = torch.tensor(self.data[2][idx + self.window_size - 1], dtype=torch.float)
        label = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)

        return signal_seq, envelope_seq, ratio_pf, label

    def is_spindle(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > THRESHOLD) else False


def get_class_idxs(dataset, distribution_mode):
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
        if is_spindle or distribution_mode == 1:
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

    def __init__(self, data_source, idx_true, idx_false, batch_size, distribution_mode, nb_batch=None):
        self.data_source = data_source
        self.idx_true = idx_true
        self.idx_false = idx_false
        self.nb_true = self.idx_true.size
        self.nb_false = self.idx_false.size
        self.length = nb_batch * batch_size if nb_batch is not None else len(self.data_source)
        self.distribution_mode = distribution_mode

    def __iter__(self):
        global precision_validation_factor
        global recall_validation_factor
        cur_iter = 0
        seed()
        #  epsilon = 1e-7
        #    proba = float(0.5 + 0.5 * (precision_validation_factor - recall_validation_factor) / (precision_validation_factor + recall_validation_factor + epsilon))
        proba = 0.5
        if self.distribution_mode == 1:
            proba = 1
        print(f"DEBUG: proba: {proba}")

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
        print(f"DEBUG: last possible for validation sampler : {self.last_possible}")

    #    self.first_idx = 0#randint(0, self.last_possible)

    def __iter__(self):
        seed(0)
        nb_iter = 5
        first_idx = [randint(0, self.last_possible)]
        while len(first_idx) < nb_iter:
            idx = randint(0, self.last_possible)
            for i in first_idx:
                if i % self.seq_stride != idx % self.seq_stride:
                    first_idx.append(idx)
        for i in range(nb_iter):
            cur_iter = 0
            cur_idx = first_idx[i]
            while cur_iter < self.length:
                cur_iter += 1
                yield cur_idx
                cur_idx += self.seq_stride

    def __len__(self):
        return self.length
        # return len(self.data_source)


def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


class ConvPoolModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 kernel_conv,
                 stride_conv,
                 conv_padding,
                 dilation_conv,
                 kernel_pool,
                 stride_pool,
                 pool_padding,
                 dilation_pool,
                 dropout_p):
        super(ConvPoolModule, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_conv,
                              stride=stride_conv,
                              padding=conv_padding,
                              dilation=dilation_conv)
        self.pool = nn.MaxPool1d(kernel_size=kernel_pool,
                                 stride=stride_pool,
                                 padding=pool_padding,
                                 dilation=dilation_pool)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return self.dropout(x)


class FcModule(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 dropout_p):
        super(FcModule, self).__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.dropout(x)


class PortiloopNetwork(nn.Module):
    def __init__(self, config_dict):
        super(PortiloopNetwork, self).__init__()

        RNN = config_dict["RNN"]
        stride_pool = config_dict["stride_pool"]
        stride_conv = config_dict["stride_conv"]
        kernel_conv = config_dict["kernel_conv"]
        kernel_pool = config_dict["kernel_pool"]
        nb_channel = config_dict["nb_channel"]
        hidden_size = config_dict["hidden_size"]
        window_size_s = config_dict["window_size_s"]
        dropout_p = config_dict["dropout"]
        dilation_conv = config_dict["dilation_conv"]
        dilation_pool = config_dict["dilation_pool"]
        fe = config_dict["fe"]
        nb_conv_layers = config_dict["nb_conv_layers"]
        nb_rnn_layers = config_dict["nb_rnn_layers"]
        first_layer_dropout = config_dict["first_layer_dropout"]
        self.envelope_input = config_dict["envelope_input"]
        self.power_features_input = config_dict["power_features_input"]

        conv_padding = 0  # int(kernel_conv // 2)
        pool_padding = 0  # int(kernel_pool // 2)
        window_size = int(window_size_s * fe)
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)

        self.RNN = RNN

        self.first_layer_input1 = ConvPoolModule(in_channels=1,
                                                 out_channel=nb_channel,
                                                 kernel_conv=kernel_conv,
                                                 stride_conv=stride_conv,
                                                 conv_padding=conv_padding,
                                                 dilation_conv=dilation_conv,
                                                 kernel_pool=kernel_pool,
                                                 stride_pool=stride_pool,
                                                 pool_padding=pool_padding,
                                                 dilation_pool=dilation_pool,
                                                 dropout_p=dropout_p if first_layer_dropout else 0)
        self.seq_input1 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
                                                         out_channel=nb_channel,
                                                         kernel_conv=kernel_conv,
                                                         stride_conv=stride_conv,
                                                         conv_padding=conv_padding,
                                                         dilation_conv=dilation_conv,
                                                         kernel_pool=kernel_pool,
                                                         stride_pool=stride_pool,
                                                         pool_padding=pool_padding,
                                                         dilation_pool=dilation_pool,
                                                         dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)

        output_cnn_size = int(nb_channel * nb_out)
        fc_size = output_cnn_size
        if RNN:
            self.gru_input1 = nn.GRU(input_size=output_cnn_size,
                                     hidden_size=hidden_size,
                                     num_layers=nb_rnn_layers,
                                     dropout=0,
                                     batch_first=True)
        #       fc_size = hidden_size
        else:
            self.first_fc_input1 = FcModule(in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
            self.seq_fc_input1 = nn.Sequential(*(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
        if self.envelope_input:
            self.first_layer_input2 = ConvPoolModule(in_channels=1,
                                                     out_channel=nb_channel,
                                                     kernel_conv=kernel_conv,
                                                     stride_conv=stride_conv,
                                                     conv_padding=conv_padding,
                                                     dilation_conv=dilation_conv,
                                                     kernel_pool=kernel_pool,
                                                     stride_pool=stride_pool,
                                                     pool_padding=pool_padding,
                                                     dilation_pool=dilation_pool,
                                                     dropout_p=dropout_p if first_layer_dropout else 0)
            self.seq_input2 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
                                                             out_channel=nb_channel,
                                                             kernel_conv=kernel_conv,
                                                             stride_conv=stride_conv,
                                                             conv_padding=conv_padding,
                                                             dilation_conv=dilation_conv,
                                                             kernel_pool=kernel_pool,
                                                             stride_pool=stride_pool,
                                                             pool_padding=pool_padding,
                                                             dilation_pool=dilation_pool,
                                                             dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))

            if RNN:
                self.gru_input2 = nn.GRU(input_size=output_cnn_size,
                                         hidden_size=hidden_size,
                                         num_layers=nb_rnn_layers,
                                         dropout=0,
                                         batch_first=True)
            else:
                self.first_fc_input2 = FcModule(in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
                self.seq_fc_input2 = nn.Sequential(*(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
        fc_features = hidden_size
        if self.envelope_input:
            fc_features += hidden_size
        if self.power_features_input:
            fc_features += 1
        out_features = 1
        if CLASSIFICATION:
            out_features = 2
        self.fc = nn.Linear(in_features=fc_features,  # enveloppe and signal + power features ratio
                            out_features=out_features)  # probability of being a spindle

    def forward(self, x1, x2, x3, h1, h2):
        (batch_size, sequence_len, features) = x1.shape
        x1 = x1.view(-1, 1, features)
        x1 = self.first_layer_input1(x1)
        x1 = self.seq_input1(x1)

        x1 = torch.flatten(x1, start_dim=1, end_dim=-1)
        hn1 = None
        if self.RNN:
            x1 = x1.view(batch_size, sequence_len, -1)
            x1, hn1 = self.gru_input1(x1, h1)
            x1 = x1[:, -1, :]
        else:
            x1 = self.first_fc_input1(x1)
            x1 = self.seq_fc_input1(x1)
        x = x1
        hn2 = None
        if self.envelope_input:
            x2 = x2.view(-1, 1, features)
            x2 = self.first_layer_input2(x2)
            x2 = self.seq_input2(x2)

            x2 = torch.flatten(x2, start_dim=1, end_dim=-1)
            if self.RNN:
                x2 = x2.view(batch_size, sequence_len, -1)
                x2, hn2 = self.gru_input2(x2, h2)
                x2 = x2[:, -1, :]
            else:
                x2 = self.first_fc_input2(x2)
                x2 = self.seq_fc_input2(x2)
            x = torch.cat((x, x2), -1)

        if self.power_features_input:
            x3 = x3.view(-1, 1)
            x = torch.cat((x, x3), -1)

        x = self.fc(x)  # output size: 1
        if CLASSIFICATION:
            x = torch.softmax(x, dim=-1)
        else:
            x = torch.sigmoid(x)
        return x, hn1, hn2


class LoggerWandb:
    def __init__(self, experiment_name, config_dict, project_name):
        self.best_model = None
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=project_name, entity="portiloop", id=experiment_name, resume="allow",
                                    config=config_dict, reinit=True)

    def log(self,
            accuracy_train,
            loss_train,
            accuracy_validation,
            loss_validation,
            f1_validation,
            precision_validation,
            recall_validation,
            best_model_accuracy_validation,
            best_epoch,
            best_model,
            loss_early_stopping,
            best_epoch_early_stopping,
            best_model_f1_score_validation,
            best_model_precision_validation,
            best_model_recall_validation,
            best_model_loss_validation,
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
        if updated_model:
            self.wandb_run.save(os.path.join(path_dataset, self.experiment_name), policy="live", base_path=path_dataset)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self):
        self.wandb_run.restore(self.experiment_name, root=path_dataset)


def get_accuracy_and_loss_pytorch(dataloader, criterion, net, device, hidden_size, nb_rnn_layers):
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.eval()
    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    loss = 0
    n = 0
    h1 = torch.zeros((nb_rnn_layers, 1, hidden_size), device=device)
    h2 = torch.zeros((nb_rnn_layers, 1, hidden_size), device=device)
    with torch.no_grad():
        for batch_data in dataloader:
            batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device).float()
            batch_samples_input2 = batch_samples_input2.to(device=device).float()
            batch_samples_input3 = batch_samples_input3.to(device=device).float()
            batch_labels = batch_labels.to(device=device).float()
            if CLASSIFICATION:
                batch_labels = (batch_labels >= THRESHOLD)
                batch_labels = batch_labels.long()
            output, h1, h2 = net_copy(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1, h2)
            if not CLASSIFICATION:
                output = output.view(-1)
            loss_py = criterion(output, batch_labels)
            loss += loss_py.item()
            if not CLASSIFICATION:
                output = (output >= THRESHOLD)
                batch_labels = (batch_labels >= THRESHOLD)
            else:
                if output.ndim == 2:
                    output = output.argmax(dim=1)

            acc += (output == batch_labels).float().mean()
            output = output.float()
            batch_labels = batch_labels.float()

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
    global precision_validation_factor
    global recall_validation_factor
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
    device_val = config_dict["device_val"]
    device_train = config_dict["device_train"]
    max_duration = config_dict["max_duration"]
    nb_rnn_layers = config_dict["nb_rnn_layers"]
    adam_w = config_dict["adam_w"]
    distribution_mode = config_dict["distribution_mode"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    logger = LoggerWandb(experiment_name, config_dict, WANDB_PROJECT)
    net = PortiloopNetwork(config_dict).to(device=device_train)
    criterion = nn.MSELoss() if not CLASSIFICATION else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr_adam, weight_decay=adam_w)

    first_epoch = 0
    try:
        logger.restore()
        checkpoint = torch.load(path_dataset / experiment_name)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch'] + 1
        recall_validation_factor = checkpoint['recall_validation_factor']
        precision_validation_factor = checkpoint['precision_validation_factor']
        print("DEBUG: Use checkpoint model")
    except (ValueError, FileNotFoundError):
        #    net = PortiloopNetwork(config_dict).to(device=device_train)
        print("DEBUG: Create new model")
    net = net.train()
    nb_weights = 0
    for i in net.parameters():
        nb_weights += len(i)
    has_envelope = 1
    if config_dict["envelope_input"]:
        has_envelope = 2
    config_dict["estimator_size_memory"] = nb_weights * window_size * seq_len * batch_size * has_envelope

    all_subject = pd.read_csv(Path(path_dataset) / "subject_sequence_p1_big.txt", header=None, delim_whitespace=True).to_numpy()
    train_subject, test_subject = train_test_split(all_subject, train_size=0.9, random_state=0)
    train_subject, validation_subject = train_test_split(train_subject, train_size=0.9, random_state=0)  # with K fold cross validation, this split will be done K times

    ds_train = SignalDataset(filename=filename_dataset,
                             path=path_dataset,
                             window_size=window_size,
                             fe=fe,
                             seq_len=seq_len,
                             seq_stride=seq_stride,
                             list_subject=train_subject)
    # start_ratio=0.0,
    # end_ratio=0.9)

    ds_validation = SignalDataset(filename=filename_dataset,
                                  path=path_dataset,
                                  window_size=window_size,
                                  fe=fe,
                                  seq_len=1,
                                  seq_stride=1,  # just to be sure, fixed value
                                  list_subject=validation_subject)
    # start_ratio=0.9,
    # end_ratio=1)

    # ds_test = SignalDataset(filename=filename, path_dataset=path_dataset, window_size=window_size, fe=fe, max_length=15, start_ratio=0.95, end_ratio=1, seq_len=1)

    idx_true, idx_false = get_class_idxs(ds_train, distribution_mode)

    samp_train = RandomSampler(ds_train,
                               idx_true=idx_true,
                               idx_false=idx_false,
                               batch_size=batch_size,
                               nb_batch=nb_batch_per_epoch,
                               distribution_mode=distribution_mode)

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

    best_model_accuracy = 0
    best_epoch = 0
    best_model = None
    best_loss_early_stopping = 1
    best_epoch_early_stopping = 0
    best_model_precision_validation = 0
    best_model_f1_score_validation = 0
    best_model_recall_validation = 0
    best_model_loss_validation = 1

    early_stopping_counter = 0
    loss_early_stopping = None
    h1_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    h2_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    for epoch in range(first_epoch, first_epoch + nb_epoch_max):

        print(f"DEBUG: epoch: {epoch}")

        accuracy_train = 0
        loss_train = 0
        n = 0
        _t_start = time.time()
        for batch_data in train_loader:
            batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device_train).float()
            batch_samples_input2 = batch_samples_input2.to(device=device_train).float()
            batch_samples_input3 = batch_samples_input3.to(device=device_train).float()
            batch_labels = batch_labels.to(device=device_train).float()

            optimizer.zero_grad()
            if CLASSIFICATION:
                batch_labels = (batch_labels >= THRESHOLD)
                batch_labels = batch_labels.long()

            output, _, _ = net(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1_zero, h2_zero)

            if not CLASSIFICATION:
                output = output.view(-1)

            loss = criterion(output, batch_labels)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()

            if not CLASSIFICATION:
                output = (output >= THRESHOLD)
                batch_labels = (batch_labels >= THRESHOLD)
            else:
                if output.ndim == 2:
                    output = output.argmax(dim=1)
            accuracy_train += (output == batch_labels).float().mean()
            n += 1
        _t_stop = time.time()
        print(f"DEBUG: Training time for 1 epoch : {_t_stop - _t_start} s")
        accuracy_train /= n
        loss_train /= n

        _t_start = time.time()
        accuracy_validation, loss_validation, f1_validation, precision_validation, recall_validation = get_accuracy_and_loss_pytorch(
            validation_loader, criterion, net, device_val, hidden_size, nb_rnn_layers)
        _t_stop = time.time()
        print(f"DEBUG: Validation time for 1 epoch : {_t_stop - _t_start} s")

        recall_validation_factor = recall_validation
        precision_validation_factor = precision_validation
        updated_model = False
        if loss_validation < best_model_loss_validation:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            # torch.save(best_model.state_dict(), path_dataset / experiment_name, _use_new_zipfile_serialization=False)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'recall_validation_factor': recall_validation_factor,
                'precision_validation_factor': precision_validation_factor,
            }, path_dataset / experiment_name, _use_new_zipfile_serialization=False)
            updated_model = True
            best_model_f1_score_validation = f1_validation
            best_model_precision_validation = precision_validation
            best_model_recall_validation = recall_validation
            best_model_loss_validation = loss_validation
            best_model_accuracy = accuracy_validation

        loss_early_stopping = max(0.2, loss_validation) if loss_early_stopping is None else loss_validation * early_stopping_smoothing_factor + loss_early_stopping * (
                1.0 - early_stopping_smoothing_factor)

        if loss_early_stopping < best_loss_early_stopping:
            best_loss_early_stopping = loss_early_stopping
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
                   best_model_accuracy_validation=best_model_accuracy,
                   best_epoch=best_epoch,
                   best_model=best_model,
                   loss_early_stopping=loss_early_stopping,
                   best_epoch_early_stopping=best_epoch_early_stopping,
                   best_model_f1_score_validation=best_model_f1_score_validation,
                   best_model_precision_validation=best_model_precision_validation,
                   best_model_recall_validation=best_model_recall_validation,
                   best_model_loss_validation=best_model_loss_validation,
                   updated_model=updated_model)

        if early_stopping_counter > nb_epoch_early_stopping_stop or time.time() - _t_start > max_duration:
            print("Early stopping.")
            break


def get_config_dict(index, name):
    config_dict = dict(experiment_name=name,
                       device_train="cuda:0",
                       device_val="cpu",
                       nb_epoch_max=1000000,
                       max_duration=int(71.5 * 3600),
                       nb_epoch_early_stopping_stop=100,
                       early_stopping_smoothing_factor=0.01,
                       fe=250,
                       nb_batch_per_epoch=1000)

    config_dict["batch_size"] = batch_size_list[index]
    config_dict["RNN"] = True
    config_dict["seq_len"] = seq_len_list[index]
    config_dict["nb_channel"] = nb_channel_list[index]
    config_dict["dropout"] = dropout_list[index]
    config_dict["hidden_size"] = hidden_size_list[index]
    config_dict["seq_stride_s"] = seq_stride_s_list[index]
    config_dict["lr_adam"] = lr_adam_list[index]
    config_dict["nb_rnn_layers"] = nb_rnn_layers_list[index]
    config_dict["first_layer_dropout"] = first_layer_dropout_list[index]
    config_dict["envelope_input"] = True
    config_dict["power_features_input"] = power_features_input_list[index]
    config_dict["time_in_past"] = config_dict["seq_len"] * config_dict["seq_stride_s"]
    config_dict["adam_w"] = adam_w_list[index]
    config_dict["distribution_mode"] = distribution_mode_list[index]

    nb_out = 0
    while nb_out < 1:
        config_dict["window_size_s"] = windows_size_s_list[index]
        config_dict["nb_conv_layers"] = nb_conv_layers_list[index]
        config_dict["stride_pool"] = stride_pool_list[index]
        config_dict["stride_conv"] = stride_conv_list[index]
        config_dict["kernel_conv"] = kernel_conv_list[index]
        config_dict["kernel_pool"] = kernel_pool_list[index]
        config_dict["dilation_conv"] = dilation_conv_list[index]
        config_dict["dilation_pool"] = dilation_pool_list[index]

        stride_pool = config_dict["stride_pool"]
        stride_conv = config_dict["stride_conv"]
        kernel_conv = config_dict["kernel_conv"]
        kernel_pool = config_dict["kernel_pool"]
        window_size_s = config_dict["window_size_s"]
        dilation_conv = config_dict["dilation_conv"]
        dilation_pool = config_dict["dilation_pool"]
        fe = config_dict["fe"]
        nb_conv_layers = config_dict["nb_conv_layers"]

        conv_padding = 0  # int(kernel_conv // 2)
        pool_padding = 0  # int(kernel_pool // 2)
        window_size = int(window_size_s * fe)
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)
    config_dict["nb_out"] = nb_out
    return config_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--experiment_index', type=int)
    args = parser.parse_args()

    exp_name = args.experiment_name
    exp_index = args.experiment_index % len(power_features_input_list)

    config_dict = get_config_dict(exp_index, exp_name)
    config_dict = {'experiment_name': 'pareto_search_8_128_v7', 'device_train': 'cuda:0', 'device_val': 'cpu', 'nb_epoch_max': 5000, 'max_duration': 257400, 'nb_epoch_early_stopping_stop': 200, 'early_stopping_smoothing_factor': 0.01, 'fe': 250, 'nb_batch_per_epoch': 10000, 'RNN': True,
                   'envelope_input': True, 'batch_size': 256, 'first_layer_dropout': False, 'power_features_input': False, 'dropout': 0.5, 'lr_adam': 0.0003, 'adam_w': 0.01, 'distribution_mode': 1, 'seq_len': 10, 'nb_channel': 43, 'hidden_size': 14, 'seq_stride_s': 0.05, 'nb_rnn_layers': 4,
                   'window_size_s': 0.08125098650113531, 'nb_conv_layers': 1, 'stride_pool': 3, 'stride_conv': 1, 'kernel_conv': 3, 'kernel_pool': 3, 'dilation_conv': 3, 'dilation_pool': 2, 'nb_out': 4, 'time_in_past': 0.5, 'estimator_size_memory': 155443200}
    run(config_dict=config_dict)
