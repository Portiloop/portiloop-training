"""
Pareto-optimal hyperparameter search (meta-learning)
"""

# all imports

from pathlib import Path
from math import floor
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
# from argparse import ArgumentParser
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle as pkl


# all constants (no hyperparameters here!)

THRESHOLD = 0.5
WANDB_PROJECT = "pareto"

filename_dataset = "dataset_big_fusion_standardized_envelope_pf.txt"

path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
path_pareto = Path(__file__).absolute().parent.parent / 'pareto'

# path = "/content/drive/MyDrive/Data/MASS/"
# path_dataset = Path(path)
# path_pareto = Path("/content/drive/MyDrive/Data/pareto_results/")

recall_validation_factor = 0.5
precision_validation_factor = 0.5

div_val_samp = 32

MAX_META_ITERATIONS = 1000  # maximum number of experiments
EPOCHS_PER_EXPERIMENT = 1  # experiments are evaluated after this number of epoch by the meta learner

EPSILON_NOISE = 0.01  # a completely random model will be selected this portion of the time, otherwise, it is sampled as a gaussian from the pareto front
ACCEPT_NOISE = 0.01  # the model will be accepted regardless of its predicted pareto-domination this portion of the time

MAX_NB_PARAMETERS = 100000  # everything over this number of parameters will be discarded

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

NB_BATCH_PER_EPOCH = 10000

RUN_NAME = "pareto_search_1"

NB_SAMPLED_MODELS_PER_ITERATION = 100  # number of models sampled per iteration, only the best predicted one is selected


# all classes and functions:

class SignalDataset(Dataset):
    def __init__(self, filename, path, window_size=64, fe=250, seq_len=5, seq_stride=5, start_ratio=0.0, end_ratio=1.0):
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename

        self.data = pd.read_csv(self.path_file, header=None).to_numpy()
        split_data = np.array(np.split(self.data, int(len(self.data) / (125 * fe))))  # 125 = nb seconds per sequence in the dataset
        np.random.seed(42)  # fixed seed value
        np.random.shuffle(split_data)
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))
        len_data = np.shape(self.data)[1]
        self.data = self.data[:, int(start_ratio * len_data):int(end_ratio * len_data)]
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

        self.labels = torch.tensor([0, 1], dtype=torch.float)

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
        global precision_validation_factor
        global recall_validation_factor
        cur_iter = 0
        seed()
        epsilon = 1e-7
        #    proba = float(0.5 + 0.5 * (precision_validation_factor - recall_validation_factor) / (precision_validation_factor + recall_validation_factor + epsilon))
        proba = 0.5
        # print(f"DEBUG: proba: {proba}")

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

    #    self.first_idx = 0#randint(0, self.last_possible)

    def __iter__(self):
        cur_iter = 0
        seed()
        self.first_idx = randint(0, self.last_possible)
        cur_idx = self.first_idx
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

        conv_padding = 0 #  int(kernel_conv // 2)
        pool_padding = 0 #  int(kernel_pool // 2)
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
        self.fc = nn.Linear(in_features=fc_features,  # enveloppe and signal + power features ratio
                            out_features=1)  # probability of being a spindle

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
        x = torch.sigmoid(x)
        return x, hn1, hn2


class LoggerWandb:
    def __init__(self, experiment_name, config_dict):
        self.best_model = None
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=WANDB_PROJECT, entity="portiloop", id=experiment_name, resume="allow",
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
        wandb.log({
            "accuracy_train": accuracy_train,
            "loss_train": loss_train,
            "accuracy_validation": accuracy_validation,
            "loss_validation": loss_validation,
            "f1_validation": f1_validation,
            "precision_validation": precision_validation,
            "recall_validation": recall_validation,
            "loss_early_stopping": loss_early_stopping,
        })
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_epoch_early_stopping"] = best_epoch_early_stopping
        wandb.run.summary["best_model_f1_score_validation"] = best_model_f1_score_validation
        wandb.run.summary["best_model_precision_validation"] = best_model_precision_validation
        wandb.run.summary["best_model_recall_validation"] = best_model_recall_validation
        wandb.run.summary["best_model_loss_validation"] = best_model_loss_validation
        wandb.run.summary["best_model_accuracy_validation"] = best_model_accuracy_validation
        if updated_model:
            wandb.run.save(os.path.join(path_dataset, self.experiment_name), policy="live", base_path=path_dataset)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self):
        wandb.run.restore(self.experiment_name, root=path_dataset)


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
            output, h1, h2 = net_copy(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1, h2)
            output = output.view(-1)
            loss_py = criterion(output, batch_labels)
            loss += loss_py.item()

            output = (output >= THRESHOLD)
            batch_labels = (batch_labels >= THRESHOLD)

            acc += (output == batch_labels).float().mean()
            output = output.float()
            batch_labels = batch_labels.float()

            # if output.ndim == 2:
            #     output = output.argmax(dim=1)

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
    # print(f"DEBUG: config_dict: {config_dict}")
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

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    # logger = LoggerWandb(experiment_name, config_dict)
    net = PortiloopNetwork(config_dict).to(device=device_train)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr_adam, weight_decay=adam_w)

    first_epoch = 0
    # try:
    #     logger.restore()
    #     checkpoint = torch.load(path_dataset / experiment_name)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     first_epoch = checkpoint['epoch'] + 1
    #     recall_validation_factor = checkpoint['recall_validation_factor']
    #     precision_validation_factor = checkpoint['precision_validation_factor']
    #     print("DEBUG: Use checkpoint model")
    # except (ValueError, FileNotFoundError):
    #     #    net = PortiloopNetwork(config_dict).to(device=device_train)
    #     print("DEBUG: Create new model")
    net = net.train()
    nb_weights = 0
    for i in net.parameters():
        nb_weights += len(i)
    has_envelope = 1
    if config_dict["envelope_input"]:
        has_envelope = 2
    config_dict["estimator_size_memory"] = nb_weights * window_size * seq_len * batch_size * has_envelope

    ds_train = SignalDataset(filename=filename_dataset,
                             path=path_dataset,
                             window_size=window_size,
                             fe=fe,
                             seq_len=seq_len,
                             seq_stride=seq_stride,
                             start_ratio=0.0,
                             end_ratio=0.9)

    ds_validation = SignalDataset(filename=filename_dataset,
                                  path=path_dataset,
                                  window_size=window_size,
                                  fe=fe,
                                  seq_len=1,
                                  start_ratio=0.9,
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

        # print(f"DEBUG: epoch: {epoch}")

        accuracy_train = 0
        loss_train = 0
        n = 0

        for batch_data in train_loader:
            batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device_train).float()
            batch_samples_input2 = batch_samples_input2.to(device=device_train).float()
            batch_samples_input3 = batch_samples_input3.to(device=device_train).float()
            batch_labels = batch_labels.to(device=device_train).float()

            optimizer.zero_grad()

            output, _, _ = net(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1_zero, h2_zero)
            output = output.view(-1)

            loss = criterion(output, batch_labels)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()

            output = output >= THRESHOLD
            batch_labels = batch_labels >= THRESHOLD

            accuracy_train += (output == batch_labels).float().mean()
            n += 1

        accuracy_train /= n
        loss_train /= n

        accuracy_validation, loss_validation, f1_validation, precision_validation, recall_validation = get_accuracy_and_loss_pytorch(
            validation_loader, criterion, net, device_val, hidden_size, nb_rnn_layers)

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

        loss_early_stopping = 1.0 if loss_early_stopping is None else loss_validation * early_stopping_smoothing_factor + loss_early_stopping * (
                1.0 - early_stopping_smoothing_factor)

        if loss_early_stopping < best_loss_early_stopping:
            best_loss_early_stopping = loss_early_stopping
            early_stopping_counter = 0
            best_epoch_early_stopping = epoch
        else:
            early_stopping_counter += 1

        # logger.log(accuracy_train=accuracy_train,
        #            loss_train=loss_train,
        #            accuracy_validation=accuracy_validation,
        #            loss_validation=loss_validation,
        #            f1_validation=f1_validation,
        #            precision_validation=precision_validation,
        #            recall_validation=recall_validation,
        #            best_model_accuracy_validation=best_model_accuracy,
        #            best_epoch=best_epoch,
        #            best_model=best_model,
        #            loss_early_stopping=loss_early_stopping,
        #            best_epoch_early_stopping=best_epoch_early_stopping,
        #            best_model_f1_score_validation=best_model_f1_score_validation,
        #            best_model_precision_validation=best_model_precision_validation,
        #            best_model_recall_validation=best_model_recall_validation,
        #            best_model_loss_validation=best_model_loss_validation,
        #            updated_model=updated_model)

        if early_stopping_counter > nb_epoch_early_stopping_stop or time.time() - _t_start > max_duration:
            print("Early stopping.")
            break

    return best_model_loss_validation


# hyperparameters

# batch_size_range_t = ["i", 256, 256]
seq_len_range_t = ["i", 20, 50]
kernel_conv_range_t = ["i", 3, 7]
kernel_pool_range_t = ["i", 3, 3]
stride_conv_range_t = ["i", 1, 1]
stride_pool_range_t = ["i", 1, 1]
dilation_conv_range_t = ["i", 1, 1]
dilation_pool_range_t = ["i", 1, 1]
nb_channel_range_t = ["i", 10, 50]
hidden_size_range_t = ["i", 2, 100]
dropout_range_t = ["f", 0.5, 0.5]
window_size_s_range_t = ["f", 0.05, 0.1]
seq_stride_s_range_t = ["f", 0.05, 0.1]
lr_adam_range_t = ["f", 0.0003, 0.0003]
nb_conv_layers_range_t = ["i", 1, 9]
nb_rnn_layers_range_t = ["i", 1, 5]
# first_layer_dropout_range_t = ["b", False, False]
# power_features_input_range_t = ["b", False, False]
adam_w_range_t = ["f", 0.0, 0.01]


def clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


def sample_from_range(range_t, gaussian_mean=None, gaussian_std_factor=1.0):
    type_t = range_t[0]
    min_t = range_t[1]
    max_t = range_t[2]
    diff_t = max_t - min_t
    gaussian_std = gaussian_std_factor * diff_t
    if type_t == "b":
        return False, False
    if gaussian_mean is None:
        res = random.uniform(min_t, max_t)
    else:
        res = random.gauss(mu=gaussian_mean, sigma=gaussian_std)
        res = clip(res, min_t, max_t)
    res_unrounded = deepcopy(res)
    if type_t == "i":
        res = round(res)
    return res, res_unrounded


def sample_config_dict(name, pareto_front):
    config_dict = dict(experiment_name=name,
                       device_train="cuda:0",
                       device_val="cpu",
                       nb_epoch_max=EPOCHS_PER_EXPERIMENT,
                       max_duration=int(71.5 * 3600),
                       nb_epoch_early_stopping_stop=200,
                       early_stopping_smoothing_factor=0.01,
                       fe=250,
                       nb_batch_per_epoch=NB_BATCH_PER_EPOCH)

    noise = random.choices(population=[True, False], weights=[EPSILON_NOISE, 1.0 - EPSILON_NOISE])

    unrounded = {}

    # constant things:

    config_dict["RNN"] = True
    config_dict["envelope_input"] = True
    config_dict["batch_size"] = 256
    config_dict["first_layer_dropout"] = False
    config_dict["power_features_input"] = False

    nb_out = 0
    while nb_out < 1:

        if len(pareto_front) == 0 or noise:
            # sample completely randomly
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t)
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t)
            config_dict["dropout"], unrounded["dropout"] = sample_from_range(dropout_range_t)
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t)
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t)
            config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t)
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t)
            config_dict["adam_w"], unrounded["adam_w"] = sample_from_range(adam_w_range_t)
            config_dict["window_size_s"], unrounded["window_size_s"] = sample_from_range(window_size_s_range_t)
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t)
            config_dict["stride_pool"], unrounded["stride_pool"] = sample_from_range(stride_pool_range_t)
            config_dict["stride_conv"], unrounded["stride_conv"] = sample_from_range(stride_conv_range_t)
            config_dict["kernel_conv"], unrounded["kernel_conv"] = sample_from_range(kernel_conv_range_t)
            config_dict["kernel_pool"], unrounded["kernel_pool"] = sample_from_range(kernel_pool_range_t)
            config_dict["dilation_conv"], unrounded["dilation_conv"] = sample_from_range(dilation_conv_range_t)
            config_dict["dilation_pool"], unrounded["dilation_pool"] = sample_from_range(dilation_pool_range_t)
        else:
            # sample gaussian from one of the previous experiments in the pareto front
            previous_experiment = random.choice(pareto_front)
            previous_unrounded = previous_experiment["unrounded"]
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t, previous_unrounded["seq_len"])
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t, previous_unrounded["nb_channel"])
            config_dict["dropout"], unrounded["dropout"] = sample_from_range(dropout_range_t, previous_unrounded["dropout"])
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t, previous_unrounded["hidden_size"])
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t, previous_unrounded["seq_stride_s"])
            config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t, previous_unrounded["lr_adam"])
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t, previous_unrounded["nb_rnn_layers"])
            config_dict["adam_w"], unrounded["adam_w"] = sample_from_range(adam_w_range_t, previous_unrounded["adam_w"])
            config_dict["window_size_s"], unrounded["window_size_s"] = sample_from_range(window_size_s_range_t, previous_unrounded["window_size_s"])
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t, previous_unrounded["nb_conv_layers"])
            config_dict["stride_pool"], unrounded["stride_pool"] = sample_from_range(stride_pool_range_t, previous_unrounded["stride_pool"])
            config_dict["stride_conv"], unrounded["stride_conv"] = sample_from_range(stride_conv_range_t, previous_unrounded["stride_conv"])
            config_dict["kernel_conv"], unrounded["kernel_conv"] = sample_from_range(kernel_conv_range_t, previous_unrounded["kernel_conv"])
            config_dict["kernel_pool"], unrounded["kernel_pool"] = sample_from_range(kernel_pool_range_t, previous_unrounded["kernel_pool"])
            config_dict["dilation_conv"], unrounded["dilation_conv"] = sample_from_range(dilation_conv_range_t, previous_unrounded["dilation_conv"])
            config_dict["dilation_pool"], unrounded["dilation_pool"] = sample_from_range(dilation_pool_range_t, previous_unrounded["dilation_pool"])

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
        config_dict["time_in_past"] = config_dict["seq_len"] * config_dict["seq_stride_s"]

    return config_dict, unrounded


def nb_parameters(config_dict):
    net = PortiloopNetwork(config_dict)
    res = sum(p.numel() for p in net.parameters())
    del net
    return res


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()

        self.fc1 = nn.Linear(in_features=16,  # nb hyperparameters
                             out_features=200)  # in SMBO paper : 25 * hyperparameters... Seems huge

        self.fc2 = nn.Linear(in_features=200,
                             out_features=200)

        self.fc3 = nn.Linear(in_features=200,
                             out_features=1)

    def to(self, device):
        super(SurrogateModel, self).to(device)
        self.device = device

    def forward(self, config_dict):
        x_list = [config_dict["seq_len"],
                  config_dict["nb_channel"],
                  config_dict["dropout"],
                  config_dict["hidden_size"],
                  config_dict["seq_stride_s"],
                  config_dict["lr_adam"],
                  config_dict["nb_rnn_layers"],
                  config_dict["adam_w"],
                  config_dict["window_size_s"],
                  config_dict["nb_conv_layers"],
                  config_dict["stride_pool"],
                  config_dict["stride_conv"],
                  config_dict["kernel_conv"],
                  config_dict["kernel_pool"],
                  config_dict["dilation_conv"],
                  config_dict["dilation_pool"]]

        x_tensor = torch.tensor(x_list).to(self.device)

        x_tensor = F.relu(self.fc1(x_tensor))
        x_tensor = F.relu(self.fc2(x_tensor))
        x_tensor = self.fc3(x_tensor)

        return x_tensor


def order_exp(exp):
  return exp["cost_software"]


def sort_pareto(pareto_front):
    pareto_front.sort(key=order_exp)
    return pareto_front


def update_pareto(experiment, pareto):
    dominates = False
    to_remove = []
    if len(pareto) == 0:
        dominates = True
    else:
        dominates = True
        for i, ep in enumerate(pareto):
            if ep["cost_software"] <= experiment["cost_software"] and ep["cost_hardware"] <= experiment["cost_hardware"]:  # remove ep from pareto
                dominates = False
            if ep["cost_software"] > experiment["cost_software"] and ep["cost_hardware"] > experiment["cost_hardware"]:  # don't remove ep from pareto
                to_remove.append(i)
    to_remove.sort(reverse=True)
    for i in to_remove:
        pareto.pop(i)
    if dominates:
        pareto.append(experiment)
    pareto = sort_pareto(pareto)
    return pareto


def dominates_pareto(experiment, pareto):
    dominates = False
    if len(pareto) == 0:
        dominates = True
    else:
        dominates = True
        for i, ep in enumerate(pareto):
            if ep["cost_software"] <= experiment["cost_software"] and ep["cost_hardware"] <= experiment["cost_hardware"]:  # remove ep from pareto
                dominates = False
    return dominates


def train_surrogate(net, all_experiments):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0, dampening=0, weight_decay=0.01, nesterov=False)
    loss = nn.MSELoss()
    losses = []
    nb_epochs = min(len(all_experiments), 100)
    for epoch in range(nb_epochs):
        random.shuffle(all_experiments)
        samples = [exp["config_dict"] for exp in all_experiments]
        labels = [exp["cost_software"] for exp in all_experiments]
        for i, sample in enumerate(samples):
            optimizer.zero_grad()
            pred = net(sample)
            targ = torch.tensor([labels[i],])
            assert pred.shape == targ.shape, f"pred.shape:{pred.shape} != targ.shape:{targ.shape}"
            l = loss(pred, targ)
            losses.append(l.item())
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
    mean_loss = np.mean(losses)
    # print(f"DEBUG: mean_loss:{mean_loss}")
    return net, mean_loss


def wandb_plot_pareto(all_experiments, ordered_pareto_front):
    # all experiments:
    x_axis = [exp["cost_hardware"] for exp in all_experiments]
    y_axis = [exp["cost_software"] for exp in all_experiments]
    plt.plot(x_axis, y_axis, 'bo')
    # pareto:
    x_axis = [exp["cost_hardware"] for exp in ordered_pareto_front]
    y_axis = [exp["cost_software"] for exp in ordered_pareto_front]
    plt.plot(x_axis, y_axis, 'ro-')
    plt.xlabel("nb parameters")
    plt.ylabel("validation loss")
    plt.draw()
    return wandb.Image(plt)


# Custom Pareto efficiency (distance from closest Pareto point)

def vector_exp(experiment):
    return np.array([experiment["cost_software"], experiment["cost_hardware"]])


def pareto_efficiency(experiment, pareto_front):
    farthest = -1.0
    v_experiment = vector_exp(experiment)
    for exp in pareto_front:
        dist = np.linalg.norm(vector_exp(exp) - v_experiment)
        if dist > farthest:
            farthest = dist
    assert farthest >= 0.0
    return farthest


def exp_max_pareto_efficiency(experiments, pareto_front):
    assert len(experiments) >= 1
    if len(pareto_front) == 0:
        return experiments[0]
    max_efficiency = 0.0
    best_exp = None
    for exp in experiments:
        efficiency = pareto_efficiency(exp, pareto_front)
        assert efficiency >= 0.0
        if efficiency >= max_efficiency:
            max_efficiency = efficiency
            best_exp = exp
    assert best_exp is not None
    return best_exp


def dump_files(all_experiments, pareto_front):
    """
    exports pickled files to path_pareto
    """
    path_current_all = path_pareto / (RUN_NAME + "_all.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    with open(path_current_all, "wb") as f:
        pkl.dump(all_experiments, f)
    with open(path_current_pareto, "wb") as f:
        pkl.dump(pareto_front, f)


def load_files():
    """
    loads pickled files from path_pareto
    returns None, None if not found
    else returns all_experiments, pareto_front
    """
    path_current_all = path_pareto / (RUN_NAME + "_all.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    if not path_current_all.exists() or not path_current_pareto.exists():
        return None, None
    with open(path_current_all, "rb") as f:
        all_experiments = pkl.load(f)
    with open(path_current_pareto, "rb") as f:
        pareto_front = pkl.load(f)
    return all_experiments, pareto_front


class LoggerWandbPareto:
    def __init__(self, run_name):
        self.run_name = run_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=WANDB_PROJECT, entity="portiloop", id=run_name, resume="allow", reinit=True)

    def log(self,
            surrogate_loss,
            surprise,
            all_experiments,
            pareto_front
            ):
        plt_img = wandb_plot_pareto(all_experiments, pareto_front)
        wandb.log({
            "surrogate_loss": surrogate_loss,
            "surprise": surprise,
            "pareto_plot": plt_img
        })

    def __del__(self):
        self.wandb_run.finish()


# Main:

if __name__ == "__main__":

    logger = LoggerWandbPareto(RUN_NAME)

    all_experiments, pareto_front = load_files()

    if all_experiments is None:
        print(f"DEBUG: no files found, starting new run")
        all_experiments = []  # list of dictionaries
        pareto_front = []  # list of dictionaries, subset of all_experiments
    else:
        print(f"DEBUG: existing run loaded")

    meta_model = SurrogateModel()
    meta_model.to(META_MODEL_DEVICE)

    # main meta-learning procedure:

    for meta_iteration in range(MAX_META_ITERATIONS):
        num_experiment = len(all_experiments)
        print("---")
        print(f"ITERATION N° {meta_iteration}")

        exp = {}
        accept_noise = random.choices(population=[True, False], weights=[ACCEPT_NOISE, 1.0 - ACCEPT_NOISE])

        exps = []

        model_selected = False
        while not model_selected:
            accept_model = False
            exp = {}
            while not accept_model:
                # sample model
                config_dict, unrounded = sample_config_dict(name=RUN_NAME + "_" + str(num_experiment), pareto_front=pareto_front)

                nb_params = nb_parameters(config_dict)
                if nb_params > MAX_NB_PARAMETERS:
                    continue

                meta_model.eval()
                with torch.no_grad():
                    predicted_loss = meta_model(config_dict).item()

                exp["cost_hardware"] = nb_params
                exp["cost_software"] = predicted_loss
                exp["config_dict"] = config_dict

                accept_model = accept_noise or dominates_pareto(exp, pareto_front)
            exps.append(exp)
            if accept_noise or len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
                # select model
                model_selected = True
                exp = exp_max_pareto_efficiency(exps, pareto_front)

        print(f"config: {config_dict}")
        # print(unrounded)

        print(f"nb parameters: {nb_params}")
        print(f"predicted loss: {predicted_loss}")
        print("training...")

        exp["cost_software"] = run(config_dict)

        pareto_front = update_pareto(exp, pareto_front)
        all_experiments.append(exp)

        print(f"actual loss: {exp['cost_software']}")
        surprise = exp['cost_software'] - predicted_loss
        print(f"surprise: {surprise}")
        print("training surrogate model...")

        meta_model.train()
        meta_model, mean_loss = train_surrogate(meta_model, all_experiments)

        print(f"surrogate model loss: {mean_loss}")

        dump_files(all_experiments, pareto_front)
        logger.log(surrogate_loss=mean_loss, surprise=surprise, all_experiments=all_experiments, pareto_front=pareto_front)

    print(f"End of meta-training.")
