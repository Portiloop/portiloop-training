"""
Main script for training an ANN.
"""

# all imports

import copy
from distutils.command.config import config
import json
import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from random import seed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from scipy.ndimage import gaussian_filter1d

from portiloop_software.portiloop_python.Utils.utils import out_dim
from portiloop_software.portiloop_python.ANN.dataset import LabelDistributionSmoothing, generate_dataloader, generate_dataloader_unlabelled_offline
from portiloop_software.portiloop_python.ANN.nn_utils import LoggerWandb, SurpriseReweighting, get_metrics
from portiloop_software.portiloop_python.ANN.configs import get_config_dict


# all classes and functions:
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

    def forward(self, input_f):
        x, max_value = input_f
        x = F.relu(self.conv(x))
        x = self.pool(x)
        max_temp = torch.max(abs(x))
        if max_temp > max_value:
            logging.debug(f"max_value = {max_temp}")
            max_value = max_temp
        return self.dropout(x), max_value


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
    def __init__(self, c_dict, exp_config):
        super(PortiloopNetwork, self).__init__()

        self.exp_config = exp_config

        RNN = c_dict["RNN"]
        stride_pool = c_dict["stride_pool"]
        stride_conv = c_dict["stride_conv"]
        kernel_conv = c_dict["kernel_conv"]
        kernel_pool = c_dict["kernel_pool"]
        nb_channel = c_dict["nb_channel"]
        hidden_size = c_dict["hidden_size"]
        window_size_s = c_dict["window_size_s"]
        dropout_p = c_dict["dropout"]
        dilation_conv = c_dict["dilation_conv"]
        dilation_pool = c_dict["dilation_pool"]
        fe = c_dict["fe"]
        nb_conv_layers = c_dict["nb_conv_layers"]
        nb_rnn_layers = c_dict["nb_rnn_layers"]
        first_layer_dropout = c_dict["first_layer_dropout"]
        self.envelope_input = c_dict["envelope_input"]
        self.power_features_input = c_dict["power_features_input"]
        self.classification = c_dict["classification"]

        conv_padding = 0  # int(kernel_conv // 2)
        pool_padding = 0  # int(kernel_pool // 2)
        window_size = int(window_size_s * fe)
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding,
                             dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding,
                             dilation_pool, kernel_pool, stride_pool)

        output_cnn_size = int(nb_channel * nb_out)

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
        if RNN:
            self.gru_input1 = nn.GRU(input_size=output_cnn_size,
                                     hidden_size=hidden_size,
                                     num_layers=nb_rnn_layers,
                                     dropout=0,
                                     batch_first=True)
        #       fc_size = hidden_size
        else:
            self.first_fc_input1 = FcModule(
                in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
            self.seq_fc_input1 = nn.Sequential(
                *(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
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
                self.first_fc_input2 = FcModule(
                    in_features=output_cnn_size, out_features=hidden_size, dropout_p=dropout_p)
                self.seq_fc_input2 = nn.Sequential(
                    *(FcModule(in_features=hidden_size, out_features=hidden_size, dropout_p=dropout_p) for _ in range(nb_rnn_layers - 1)))
        fc_features = 0
        fc_features += hidden_size
        if self.envelope_input:
            fc_features += hidden_size
        if self.power_features_input:
            fc_features += 1
        out_features = 1
        self.fc = nn.Linear(in_features=fc_features,  # enveloppe and signal + power features ratio
                            out_features=out_features)  # probability of being a spindle

    def forward(self, x1, x2, x3, h1, h2, max_value=np.inf):
        # x1 : input 1 : cleaned signal
        # x2 : input 2 : envelope
        # x3 : power features ratio
        # h1 : gru 1 hidden size
        # h2 : gru 2 hidden size
        # max_value (optional) : print the maximal value reach during inference (used to verify if the FPGA implementation precision is enough)
        (batch_size, sequence_len, features) = x1.shape

        if self.exp_config['ablation'] == 1:
            x1 = copy.deepcopy(x2)
        elif self.exp_config['ablation'] == 2:
            x2 = copy.deepcopy(x1)

        x1 = x1.view(-1, 1, features)
        x1, max_value = self.first_layer_input1((x1, max_value))
        x1, max_value = self.seq_input1((x1, max_value))

        x1 = torch.flatten(x1, start_dim=1, end_dim=-1)
        hn1 = None
        if self.RNN:
            x1 = x1.view(batch_size, sequence_len, -1)
            x1, hn1 = self.gru_input1(x1, h1)
            max_temp = torch.max(abs(x1))
            if max_temp > max_value:
                logging.debug(f"max_value = {max_temp}")
                max_value = max_temp
            x1 = x1[:, -1, :]
        else:
            x1 = self.first_fc_input1(x1)
            x1 = self.seq_fc_input1(x1)
        x = x1
        hn2 = None
        if self.envelope_input:
            x2 = x2.view(-1, 1, features)
            x2, max_value = self.first_layer_input2((x2, max_value))
            x2, max_value = self.seq_input2((x2, max_value))

            x2 = torch.flatten(x2, start_dim=1, end_dim=-1)
            if self.RNN:
                x2 = x2.view(batch_size, sequence_len, -1)
                x2, hn2 = self.gru_input2(x2, h2)
                max_temp = torch.max(abs(x2))
                if max_temp > max_value:
                    logging.debug(f"max_value = {max_temp}")
                    max_value = max_temp
                x2 = x2[:, -1, :]
            else:
                x2 = self.first_fc_input2(x2)
                x2 = self.seq_fc_input2(x2)
            x = torch.cat((x, x2), -1)

        if self.power_features_input:
            x3 = x3.view(-1, 1)
            x = torch.cat((x, x3), -1)

        x = self.fc(x)  # output size: 1
        max_temp = torch.max(abs(x))
        if max_temp > max_value:
            logging.debug(f"max_value = {max_temp}")
            max_value = max_temp
        x = torch.sigmoid(x)

        return x, hn1, hn2, max_value


def run_inference(exp_config, dataloader, criterion, net, device, hidden_size, nb_rnn_layers, classification, batch_size_validation, max_value=np.inf):
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.eval()
    loss = 0
    n = 0
    batch_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)
    h1 = torch.zeros((nb_rnn_layers, batch_size_validation,
                     hidden_size), device=device)
    h2 = torch.zeros((nb_rnn_layers, batch_size_validation,
                     hidden_size), device=device)
    with torch.no_grad():
        for batch_data in dataloader:
            batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(
                device=device).float()
            batch_samples_input2 = batch_samples_input2.to(
                device=device).float()
            batch_samples_input3 = batch_samples_input3.to(
                device=device).float()
            batch_labels = batch_labels.to(device=device).float()
            if classification:
                batch_labels = (batch_labels > exp_config['threshold'])
                batch_labels = batch_labels.float()
            output, h1, h2, max_value = net_copy(
                batch_samples_input1, batch_samples_input2, batch_samples_input3, h1, h2, max_value)
            # logging.debug(f"label = {batch_labels}")
            # logging.debug(f"output = {output}")
            output = output.view(-1)
            loss_py = criterion(output, batch_labels).mean()
            loss += loss_py.item()
            # logging.debug(f"loss = {loss}")
            if not classification:
                output = (output > exp_config['threshold'])
                batch_labels = (batch_labels > exp_config['threshold'])
            else:
                output = (output >= 0.5)
            batch_labels_total = torch.cat([batch_labels_total, batch_labels])
            output_total = torch.cat([output_total, output])
            # logging.debug(f"batch_label_total : {batch_labels_total}")
            # logging.debug(f"output_total : {output_total}")
            n += 1

    loss /= n
    acc = (output_total == batch_labels_total).float().mean()
    output_total = output_total.float()
    batch_labels_total = batch_labels_total.float()
    tp = (batch_labels_total * output_total)
    tn = ((1 - batch_labels_total) * (1 - output_total))
    fp = ((1 - batch_labels_total) * output_total)
    fn = (batch_labels_total * (1 - output_total))
    return output_total, batch_labels_total, loss, acc, tp, tn, fp, fn


def run_inference_unlabelled_offline(dataloader, net, device, hidden_size, nb_rnn_layers, classification, batch_size_validation):
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.eval()
    true_idx_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)
    h1 = torch.zeros((nb_rnn_layers, batch_size_validation,
                     hidden_size), device=device)
    h2 = torch.zeros((nb_rnn_layers, batch_size_validation,
                     hidden_size), device=device)
    max_value = np.inf
    with torch.no_grad():
        for batch_data in dataloader:
            batch_samples_input1, batch_true_idx = batch_data
            batch_samples_input1 = batch_samples_input1.to(
                device=device).float()
            output, h1, h2, max_value = net_copy(
                batch_samples_input1, None, None, h1, h2, max_value)
            output = output.view(-1)
            # if not classification:
            #     output = output  # (output > THRESHOLD)
            # else:
            #     output = (output >= 0.5)
            true_idx_total = torch.cat([true_idx_total, batch_true_idx])
            output_total = torch.cat([output_total, output])
    output_total = output_total.float()
    true_idx_total = true_idx_total.int()
    return output_total, true_idx_total


def run(nn_config, data_config, exp_config, wandb_project, save_model, unique_name):
    global precision_validation_factor
    global recall_validation_factor
    _t_start = time.time()
    logging.debug(f"config_dict: {nn_config}")
    experiment_name = f"{nn_config['experiment_name']}_{time.time_ns()}" if unique_name else nn_config['experiment_name']
    nb_epoch_max = nn_config["nb_epoch_max"]
    nb_batch_per_epoch = nn_config["nb_batch_per_epoch"]
    nb_epoch_early_stopping_stop = nn_config["nb_epoch_early_stopping_stop"]
    early_stopping_smoothing_factor = nn_config["early_stopping_smoothing_factor"]
    batch_size = nn_config["batch_size"]
    seq_len = nn_config["seq_len"]
    window_size_s = nn_config["window_size_s"]
    fe = nn_config["fe"]
    seq_stride_s = nn_config["seq_stride_s"]
    lr_adam = nn_config["lr_adam"]
    hidden_size = nn_config["hidden_size"]
    device_val = nn_config["device_val"]
    device_train = nn_config["device_train"]
    max_duration = nn_config["max_duration"]
    nb_rnn_layers = nn_config["nb_rnn_layers"]
    adam_w = nn_config["adam_w"]
    distribution_mode = nn_config["distribution_mode"]
    classification = nn_config["classification"]
    reg_balancing = nn_config["reg_balancing"]
    split_idx = nn_config["split_idx"]
    validation_network_stride = nn_config["validation_network_stride"]

    assert reg_balancing in {'none', 'lds',
                             'sr'}, f"wrong key: {reg_balancing}"
    assert classification or distribution_mode == 1, "distribution_mode must be 1 (no class balancing) in regression mode"
    balancer_type = 0
    lds = None
    sr = None
    if reg_balancing == 'lds':
        balancer_type = 1
    elif reg_balancing == 'sr':
        balancer_type = 2

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    logger = LoggerWandb(experiment_name, nn_config, wandb_project)
    torch.seed()
    net = PortiloopNetwork(nn_config).to(device=device_train)
    criterion = nn.MSELoss(
        reduction='none') if not classification else nn.BCELoss(reduction='none')
    # criterion = nn.MSELoss() if not classification else nn.BCELoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr_adam, weight_decay=adam_w)
    best_loss_early_stopping = 1
    best_epoch_early_stopping = 0
    best_model_precision_validation = 0
    best_model_f1_score_validation = 0
    best_model_recall_validation = 0
    best_model_loss_validation = 1

    best_model_on_loss_accuracy = 0
    best_model_on_loss_precision_validation = 0
    best_model_on_loss_f1_score_validation = 0
    best_model_on_loss_recall_validation = 0
    best_model_on_loss_loss_validation = 1

    first_epoch = 0
    try:
        logger.restore(classification)
        file_exp = experiment_name
        file_exp += "" if classification else "_on_loss"
        if not device_val.startswith("cuda"):
            checkpoint = torch.load(
                data_config['path_dataset'] / file_exp, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(data_config['path_dataset'] / file_exp)
        logging.debug("Use checkpoint model")
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch'] + 1
        recall_validation_factor = checkpoint['recall_validation_factor']
        precision_validation_factor = checkpoint['precision_validation_factor']
        best_model_on_loss_loss_validation = checkpoint['best_model_on_loss_loss_validation']
        best_model_f1_score_validation = checkpoint['best_model_f1_score_validation']
    except (ValueError, FileNotFoundError):
        #    net = PortiloopNetwork(config_dict).to(device=device_train)
        logging.debug("Create new model")
    net = net.train()
    nb_weights = 0
    for i in net.parameters():
        nb_weights += len(i)
    has_envelope = 1
    if nn_config["envelope_input"]:
        has_envelope = 2
    nn_config["estimator_size_memory"] = nb_weights * \
        window_size * seq_len * batch_size * has_envelope

    train_loader, validation_loader, batch_size_validation, _, _, _ = generate_dataloader(window_size, fe, seq_len, seq_stride, distribution_mode,
                                                                                          batch_size, nb_batch_per_epoch, classification, split_idx,
                                                                                          validation_network_stride)
    if balancer_type == 1:
        lds = LabelDistributionSmoothing(c=1.0, dataset=train_loader.dataset, weights=None, kernel_size=5, kernel_std=0.01, nb_bins=100,
                                         weighting_mode='inv_sqrt')
    elif balancer_type == 2:
        sr = SurpriseReweighting(weights=None, nb_bins=100, alpha=1e-3)

    best_model_accuracy = 0
    best_epoch = 0
    best_model = None

    accuracy_train = None
    loss_train = None

    early_stopping_counter = 0
    loss_early_stopping = None
    h1_zero = torch.zeros(
        (nb_rnn_layers, batch_size, hidden_size), device=device_train)
    h2_zero = torch.zeros(
        (nb_rnn_layers, batch_size, hidden_size), device=device_train)
    for epoch in range(first_epoch, first_epoch + nb_epoch_max):

        logging.debug(f"epoch: {epoch}")

        n = 0
        if epoch > -1:
            accuracy_train = 0
            loss_train = 0
            _t_start = time.time()
            for batch_data in train_loader:
                batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
                batch_samples_input1 = batch_samples_input1.to(
                    device=device_train).float()
                batch_samples_input2 = batch_samples_input2.to(
                    device=device_train).float()
                batch_samples_input3 = batch_samples_input3.to(
                    device=device_train).float()
                batch_labels = batch_labels.to(device=device_train).float()

                optimizer.zero_grad()
                if classification:
                    batch_labels = (batch_labels > exp_config['threshold'])
                    batch_labels = batch_labels.float()

                output, _, _, _ = net(
                    batch_samples_input1, batch_samples_input2, batch_samples_input3, h1_zero, h2_zero)

                output = output.view(-1)

                loss = criterion(output, batch_labels)

                if balancer_type == 1:
                    batch_weights = lds.lds_weights_batch(batch_labels)
                    loss = loss * batch_weights
                    error = batch_weights.isinf().any().item() or batch_weights.isnan().any().item() or torch.isnan(
                        loss).any().item() or torch.isinf(loss).any().item()
                    if error:
                        logging.debug(f"batch_labels: {batch_labels}")
                        logging.debug(f"batch_weights: {batch_weights}")
                        logging.debug(f"loss: {loss}")
                        logging.debug(f"LDS: {lds}")
                        assert False, "loss is nan or inf"
                elif balancer_type == 2:
                    loss = sr.update_and_get_weighted_loss(
                        batch_labels=batch_labels, unweighted_loss=loss)
                    error = torch.isnan(loss).any().item(
                    ) or torch.isinf(loss).any().item()
                    if error:
                        logging.debug(f"batch_labels: {batch_labels}")
                        logging.debug(f"loss: {loss}")
                        logging.debug(f"SR: {sr}")
                        assert False, "loss is nan or inf"

                loss = loss.mean()

                loss_train += loss.item()
                loss.backward()
                optimizer.step()

                if not classification:
                    output = (output > exp_config['threshold'])
                    batch_labels = (batch_labels > exp_config['threshold'])
                else:
                    output = (output >= 0.5)
                accuracy_train += (output == batch_labels).float().mean()
                n += 1
            _t_stop = time.time()
            logging.debug(
                f"Training time for 1 epoch : {_t_stop - _t_start} s")
            accuracy_train /= n
            loss_train /= n

            _t_start = time.time()
        output_validation, labels_validation, loss_validation, accuracy_validation, tp, tn, fp, fn = run_inference(exp_config, validation_loader, criterion, net,
                                                                                                                   device_val, hidden_size,
                                                                                                                   nb_rnn_layers, classification,
                                                                                                                   batch_size_validation)
        f1_validation, precision_validation, recall_validation = get_metrics(
            tp, fp, fn)

        _t_stop = time.time()
        logging.debug(f"Validation time for 1 epoch : {_t_stop - _t_start} s")

        recall_validation_factor = recall_validation
        precision_validation_factor = precision_validation
        updated_model = False
        if f1_validation > best_model_f1_score_validation:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            # torch.save(best_model.state_dict(), path_dataset / experiment_name, _use_new_zipfile_serialization=False)
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_validation_factor': recall_validation_factor,
                    'precision_validation_factor': precision_validation_factor,
                    'best_model_on_loss_loss_validation': best_model_on_loss_loss_validation,
                    'best_model_f1_score_validation': best_model_f1_score_validation,
                }, data_config['path_dataset'] / experiment_name, _use_new_zipfile_serialization=False)
                updated_model = True
            best_model_f1_score_validation = f1_validation
            best_model_precision_validation = precision_validation
            best_model_recall_validation = recall_validation
            best_model_loss_validation = loss_validation
            best_model_accuracy = accuracy_validation
        if loss_validation < best_model_on_loss_loss_validation:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            # torch.save(best_model.state_dict(), path_dataset / experiment_name, _use_new_zipfile_serialization=False)
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_validation_factor': recall_validation_factor,
                    'precision_validation_factor': precision_validation_factor,
                    'best_model_on_loss_loss_validation': best_model_on_loss_loss_validation,
                    'best_model_f1_score_validation': best_model_f1_score_validation,
                }, data_config['path_dataset'] / (experiment_name + "_on_loss"), _use_new_zipfile_serialization=False)
                updated_model = True
            best_model_on_loss_f1_score_validation = f1_validation
            best_model_on_loss_precision_validation = precision_validation
            best_model_on_loss_recall_validation = recall_validation
            best_model_on_loss_loss_validation = loss_validation
            best_model_on_loss_accuracy = accuracy_validation

        loss_early_stopping = loss_validation if loss_early_stopping is None and early_stopping_smoothing_factor == 1 else loss_validation if loss_early_stopping is None else loss_validation * early_stopping_smoothing_factor + loss_early_stopping * (
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
                   best_epoch=best_epoch,
                   best_model=best_model,
                   loss_early_stopping=loss_early_stopping,
                   best_epoch_early_stopping=best_epoch_early_stopping,
                   best_model_accuracy_validation=best_model_accuracy,
                   best_model_f1_score_validation=best_model_f1_score_validation,
                   best_model_precision_validation=best_model_precision_validation,
                   best_model_recall_validation=best_model_recall_validation,
                   best_model_loss_validation=best_model_loss_validation,
                   best_model_on_loss_accuracy_validation=best_model_on_loss_accuracy,
                   best_model_on_loss_f1_score_validation=best_model_on_loss_f1_score_validation,
                   best_model_on_loss_precision_validation=best_model_on_loss_precision_validation,
                   best_model_on_loss_recall_validation=best_model_on_loss_recall_validation,
                   best_model_on_loss_loss_validation=best_model_on_loss_loss_validation,
                   updated_model=updated_model)

        if early_stopping_counter > nb_epoch_early_stopping_stop or time.time() - _t_start > max_duration:
            logging.debug("Early stopping.")
            break
    logging.debug("Delete logger")
    del logger
    logging.debug("Logger deleted")
    return best_model_loss_validation, best_model_f1_score_validation, best_epoch_early_stopping


def run_offline_unlabelled(config_dict, path_experiments, unlabelled_segment):
    logging.debug(f"config_dict: {config_dict}")
    experiment_name = config_dict['experiment_name']
    window_size_s = config_dict["window_size_s"]
    fe = config_dict["fe"]
    seq_stride_s = config_dict["seq_stride_s"]
    hidden_size = config_dict["hidden_size"]
    device_inference = config_dict["device_inference"]
    nb_rnn_layers = config_dict["nb_rnn_layers"]
    classification = config_dict["classification"]
    validation_network_stride = config_dict["validation_network_stride"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_inference.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    torch.seed()
    net = PortiloopNetwork(config_dict).to(device=device_inference)

    file_exp = experiment_name
    file_exp += "" if classification else "_on_loss"
    path_experiments = Path(path_experiments)
    if not device_inference.startswith("cuda"):
        checkpoint = torch.load(
            path_experiments / file_exp, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_experiments / file_exp)
    logging.debug("Use checkpoint model")
    net.load_state_dict(checkpoint['model_state_dict'])

    test_loader, batch_size_test = generate_dataloader_unlabelled_offline(unlabelled_segment=unlabelled_segment,
                                                                          window_size=window_size,
                                                                          seq_stride=seq_stride,
                                                                          network_stride=validation_network_stride)

    output_total, true_idx_total = run_inference_unlabelled_offline(dataloader=test_loader,
                                                                    net=net,
                                                                    device=device_inference,
                                                                    hidden_size=hidden_size,
                                                                    nb_rnn_layers=nb_rnn_layers,
                                                                    classification=classification,
                                                                    batch_size_validation=batch_size_test)
    return output_total, true_idx_total


def initialize_nn_config(experiment_name):
    config_dict = {
        'experiment_name': experiment_name,
        'device_train': 'cuda:0',
        'device_val': 'cuda:0',
        'nb_epoch_max': 11,
        'max_duration': 257400,
        'nb_epoch_early_stopping_stop': 10,
        'early_stopping_smoothing_factor': 0.1,
        'fe': 250,
        'nb_batch_per_epoch': 5000,
        'batch_size': 256,
        'first_layer_dropout': False,
        'power_features_input': False,
        'dropout': 0.5,
        'adam_w': 0.01,
        'distribution_mode': 0,
        'classification': True,
        'nb_conv_layers': 3,
        'seq_len': 50,
        'nb_channel': 16,
        'hidden_size': 32,
        'seq_stride_s': 0.08600000000000001,
        'nb_rnn_layers': 1,
        'RNN': True,
        'envelope_input': True,
        'lr_adam': 0.0007,
        'window_size_s': 0.266,
        'stride_pool': 1,
        'stride_conv': 1,
        'kernel_conv': 9,
        'kernel_pool': 7,
        'dilation_conv': 1,
        'dilation_pool': 1,
        'nb_out': 24,
        'time_in_past': 4.300000000000001,
        'estimator_size_memory': 1628774400}
    with open(f"{experiment_name}_nn_config.json", "w") as outfile:
        json.dump(config_dict, outfile)
    return config_dict


def initialize_dataset_config(phase, path_dataset=None, reg_filename=None, class_filename=None, sl=None, sl_p1=None, sl_p2=None):
    # Initialize with default filename
    if path_dataset is None:
        path_dataset = Path(__file__).absolute(
        ).parent.parent.parent / 'dataset'

    if reg_filename is None:
        reg_filename = f"dataset_regression_{phase}_big_250_matlab_standardized_envelope_pf.txt"
    if class_filename is None:
        class_filename = f"dataset_classification_{phase}_big_250_matlab_standardized_envelope_pf.txt"
    if sl is None:
        sl = f"subject_sequence_{phase}_big.txt"
    if sl_p1 is None:
        sl_p1 = f"subject_sequence_p1_big.txt"
    if sl_p2 is None:
        sl_p2 = f"subject_sequence_p2_big.txt"

    return {'path_dataset': path_dataset,
            'filename_regression_dataset': reg_filename,
            'filename_classification_dataset': class_filename,
            'subject_list': sl,
            'subject_list_p1': sl_p1,
            'subject_list_p2': sl_p2}


if __name__ == "__main__":

    recall_validation_factor = 0.5
    precision_validation_factor = 0.5
    # TODO : Something with this

    # Parser definition
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment name for Wandb')
    parser.add_argument('--experiment_index', type=int,
                        help='Experiment index for Wandb')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--phase', type=str, default='full')
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--max_split', type=int, default=10)
    parser.add_argument('--config_nn', type=str, default=None)
    parser.add_argument('--config_data', type=str, default=None)
    # Group for test set argument
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument(
        '--test_set', dest='test_set', action='store_true')
    feature_parser.add_argument(
        '--no_test_set', dest='test_set', action='store_false')
    parser.set_defaults(test_set=True)
    # Group for classification argument
    feature_class_parser = parser.add_mutually_exclusive_group(required=False)
    feature_class_parser.add_argument(
        '--classification', dest='classification', action='store_true')
    feature_class_parser.add_argument(
        '--regression', dest='classification', action='store_false')
    parser.set_defaults(classification=True)
    args = parser.parse_args()

    # Initialize configuration dictionary for dataset
    if args.config_data is None:
        data_config = initialize_dataset_config(
            args.phase, path_dataset=Path(args.path) if args.path is not None else None)
    else:
        data_config = json.loads(args.config_data)

    # initialize output file for logging
    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s',
                            filename=args.output_file, level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # Initialize configuration dictionary for NN
    if args.config_nn is None:
        # Option with some random hyperparameters
        nn_config = initialize_nn_config()
    else:
        try:
            # Read from json file
            nn_config = json.loads(args.config_nn)
        except Exception:
            # REad from one of defaults
            exp_index = args.experiment_index
            possible_split = [0, 2]
            split_idx = possible_split[exp_index % 2]
            nn_config = get_config_dict(
                args.config_nn, args.ablation, exp_index, split_idx)

    # Set classification mode and name of experiment
    nn_config['distribution_mode'] = 0 if args.classification else 1
    nn_config['classification'] = args.classification
    nn_config['experiment_name'] += "_regression" if not args.classification else ""
    nn_config['experiment_name'] += "_no_test" if not args.test_set else ""

    # Initialize a dictionary with all hyperparameters of the experiment
    threshold_list = {'p1': 0.2, 'p2': 0.35, 'full': 0.2}  # full = p1 + p2
    exp_config = {
        'ablation': args.ablation,  # 0 : no ablation, 1 : remove input 1, 2 : remove input 2
        'phase': args.phase,
        'threshold': threshold_list[args.phase],
        'len_segment': 115
    }

    logging.debug(f"classification: {args.classification}")

    seed()  # reset the seed

    # Start the run
    run(nn_config=nn_config, data_config=data_config, exp_config=exp_config, wandb_project=f"{args.phase}-dataset-public",
        save_model=True, unique_name=False)
