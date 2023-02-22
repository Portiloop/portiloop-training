"""
Main script for training an ANN.
"""

# all imports

import copy
import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from random import randint, seed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import wandb
from portiloop_software.portiloop_python.ANN.data.data import generate_dataloader, generate_dataloader_unlabelled_offline
from portiloop_software.portiloop_python.ANN.data.reg_balancing import LabelDistributionSmoothing, SurpriseReweighting

from portiloop_software.portiloop_python.ANN.models.lstm import PortiloopNetwork

from portiloop_software.portiloop_python.Utils.utils import out_dim

recall_validation_factor = 0.5
precision_validation_factor = 0.5

# hyperparameters

# batch_size_list = [64, 64, 64, 128, 128, 128, 256, 256, 256]
# lr_adam_list = [0.0003, 0.0005, 0.0009]
# hidden_size_list = [2, 5, 10, 15, 20]

LEN_SEGMENT = 115  # in seconds
PHASE = "full"

# all classes and functions:

class LoggerWandb:
    def __init__(self, experiment_name, c_dict, project_name):
        self.best_model = None
        self.experiment_name = experiment_name
        os.environ['WANDB_API_KEY'] = "cd105554ccdfeee0bbe69c175ba0c14ed41f6e00"
        self.wandb_run = wandb.init(project=project_name, entity="portiloop", id=experiment_name, resume="allow",
                                    config=c_dict, reinit=True)
        self.c_dict = c_dict


    def log(self,
            accuracy_train,
            loss_train,
            accuracy_validation,
            loss_validation,
            f1_validation,
            precision_validation,
            recall_validation,
            best_epoch,
            best_model,
            loss_early_stopping,
            best_epoch_early_stopping,
            best_model_accuracy_validation,
            best_model_f1_score_validation,
            best_model_precision_validation,
            best_model_recall_validation,
            best_model_loss_validation,
            best_model_on_loss_accuracy_validation,
            best_model_on_loss_f1_score_validation,
            best_model_on_loss_precision_validation,
            best_model_on_loss_recall_validation,
            best_model_on_loss_loss_validation,
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
        self.wandb_run.summary["best_model_on_loss_f1_score_validation"] = best_model_on_loss_f1_score_validation
        self.wandb_run.summary["best_model_on_loss_precision_validation"] = best_model_on_loss_precision_validation
        self.wandb_run.summary["best_model_on_loss_recall_validation"] = best_model_on_loss_recall_validation
        self.wandb_run.summary["best_model_on_loss_loss_validation"] = best_model_on_loss_loss_validation
        self.wandb_run.summary["best_model_on_loss_accuracy_validation"] = best_model_on_loss_accuracy_validation
        # if updated_model:
        #     self.wandb_run.save(os.path.join(path_dataset, self.experiment_name), policy="live", base_path=path_dataset)
        #     self.wandb_run.save(os.path.join(path_dataset, self.experiment_name + "_on_loss"), policy="live", base_path=path_dataset)

    def __del__(self):
        self.wandb_run.finish()

    def restore(self, classif):
        if classif:
            self.wandb_run.restore(self.experiment_name, root=self.c_dict['path_dataset'])
        else:
            self.wandb_run.restore(self.experiment_name + "_on_loss", root=self.c_dict['path_dataset'])


def f1_loss(output, batch_labels):
    # logging.debug(f"output in loss : {output[:,1]}")
    # logging.debug(f"batch_labels in loss : {batch_labels}")
    y_pred = output
    tp = (batch_labels * y_pred).sum().to(torch.float32)
    tn = ((1 - batch_labels) * (1 - y_pred)).sum().to(torch.float32).item()
    fp = ((1 - batch_labels) * y_pred).sum().to(torch.float32)
    fn = (batch_labels * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    F1_class1 = 2 * tp / (2 * tp + fp + fn + epsilon)
    F1_class0 = 2 * tn / (2 * tn + fn + fp + epsilon)
    New_F1 = (F1_class1 + F1_class0) / 2
    return 1 - New_F1


def run_adaptation(dataloader, net, device, config):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(net.parameters(), lr=config['lr_adam'], weight_decay=config['adam_w'])
    criterion = nn.BCELoss(reduction='none')

    # Initialize All the necessary variables
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.train()
    loss = 0
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1, config['hidden_size']), device=device)

    # Run through the dataloader
    out_grus = []
    out_loss = 0
    for index, info in enumerate(dataloader):
        # Get the data and labels
        window_data, window_labels = info
        window_data = window_data.to(device)
        window_labels = window_labels.to(device)

        optimizer.zero_grad()

        # Get the output of the network
        output, h1, out_gru = net_copy(window_data, h1, torch.tensor(out_grus))
        out_grus.append(out_gru)

        if len(out_grus) > config['max_h_length']:
            out_grus.pop(0)

        # Compute the loss
        loss = criterion(output, window_labels)

        if index % 1 == 0:
            # Update the model
            loss.backward()
            optimizer.step()

        # Update the loss
        out_loss += loss.item()
        n += 1

        # Get the predictions
        output = (output >= 0.5)

        # Concatenate the predictions
        window_labels_total = torch.cat([window_labels_total, window_labels])
        output_total = torch.cat([output_total, output])
    
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


def run_inference(dataloader, criterion, net, device, hidden_size, nb_rnn_layers, classification, batch_size_validation, threshold):
    """
    Runs a validation inference over a whole dataset and returns the loss and accuracy.
    Aslo returns fp, fn, tp, tn count for spindles
    """
    
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.eval()
    loss = 0
    n = 0
    batch_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero 
    h1 = torch.zeros((nb_rnn_layers, batch_size_validation, hidden_size), device=device)

    # Run through the dataloader
    with torch.no_grad():
        out_grus = []
        for batch_data in dataloader:
            # Get the current batch data
            batch_samples_input1, _, _, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device).float()
            batch_labels = batch_labels.to(device=device).float()
            if classification:
                batch_labels = (batch_labels > threshold)
                batch_labels = batch_labels.float()

            # Run the model
            output, h1, out_gru = net_copy(batch_samples_input1, h1)
            out_grus.append(out_gru)
            MAX_H_SIZE = 100
            if len(out_grus) > MAX_H_SIZE:
                out_grus.pop(0)

            # Compute the loss
            output = output.view(-1)
            loss_py = criterion(output, batch_labels).mean()
            loss += loss_py.item()

            # Get the predictions
            if not classification:
                output = (output > threshold)
                batch_labels = (batch_labels > threshold)
            else:
                output = (output >= 0.5)

            # Concatenate the predictions
            batch_labels_total = torch.cat([batch_labels_total, batch_labels])
            output_total = torch.cat([output_total, output])
            n += 1

    # Compute metrics
    loss /= n
    acc = (output_total == batch_labels_total).float().mean()
    output_total = output_total.float()
    batch_labels_total = batch_labels_total.float()
    # Get the true positives, true negatives, false positives and false negatives
    tp = (batch_labels_total * output_total)
    tn = ((1 - batch_labels_total) * (1 - output_total))
    fp = ((1 - batch_labels_total) * output_total)
    fn = (batch_labels_total * (1 - output_total))

    return output_total, batch_labels_total, loss, acc, tp, tn, fp, fn


def run_inference_unlabelled_offline(dataloader, net, device, hidden_size, nb_rnn_layers, classification, batch_size_validation):
    """
    Simply run inference on an unlabelled dataset
    """
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.eval()
    true_idx_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)
    h1 = torch.zeros((nb_rnn_layers, batch_size_validation, hidden_size), device=device)
    h2 = torch.zeros((nb_rnn_layers, batch_size_validation, hidden_size), device=device)
    max_value = np.inf
    with torch.no_grad():
        for batch_data in dataloader:
            batch_samples_input1, batch_true_idx = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device).float()
            output, h1, h2, max_value = net_copy(batch_samples_input1, None, None, h1, h2, max_value)
            output = output.view(-1)
            if not classification:
                output = output  # (output > THRESHOLD)
            else:
                output = (output >= 0.5)
            true_idx_total = torch.cat([true_idx_total, batch_true_idx])
            output_total = torch.cat([output_total, output])
    output_total = output_total.float()
    true_idx_total = true_idx_total.int()
    return output_total, true_idx_total


def get_metrics(tp, fp, fn):
    """
    Compute the F1, precision and recall for spindles from a true positive count, false positive count and false negative count
    """
    tp_sum = tp.sum().to(torch.float32).item()
    fp_sum = fp.sum().to(torch.float32).item()
    fn_sum = fn.sum().to(torch.float32).item()
    epsilon = 1e-7

    precision = tp_sum / (tp_sum + fp_sum + epsilon)
    recall = tp_sum / (tp_sum + fn_sum + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall
    

def run(config_dict, wandb_project, save_model, unique_name):
    global precision_validation_factor
    global recall_validation_factor
    _t_start = time.time()
    logging.debug(f"config_dict: {config_dict}")
    experiment_name = f"{config_dict['experiment_name']}_{time.time_ns()}" if unique_name else config_dict['experiment_name']
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
    classification = config_dict["classification"]
    reg_balancing = config_dict["reg_balancing"]
    split_idx = config_dict["split_idx"]
    validation_network_stride = config_dict["validation_network_stride"]

    assert reg_balancing in {'none', 'lds', 'sr'}, f"wrong key: {reg_balancing}"
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

    logger = LoggerWandb(experiment_name, config_dict, wandb_project)
    torch.seed()
    net = PortiloopNetwork(config_dict).to(device=device_train)
    criterion = nn.MSELoss(reduction='none') if not classification else nn.BCELoss(reduction='none')
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
        checkpoint = torch.load(config_dict['path_dataset'] / file_exp)
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
    if config_dict["envelope_input"]:
        has_envelope = 2
    config_dict["estimator_size_memory"] = nb_weights * window_size * seq_len * batch_size * has_envelope

    train_loader, validation_loader, batch_size_validation, _, _, _ = generate_dataloader(config_dict)
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
    h1_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    h2_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    for epoch in range(first_epoch, first_epoch + nb_epoch_max):

        logging.debug(f"epoch: {epoch}")

        n = 0
        if epoch > -1:
            accuracy_train = 0
            loss_train = 0
            _t_start = time.time()
            for batch_data in train_loader:
                batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = batch_data
                batch_samples_input1 = batch_samples_input1.to(device=device_train).float()
                batch_samples_input2 = batch_samples_input2.to(device=device_train).float()
                batch_samples_input3 = batch_samples_input3.to(device=device_train).float()
                batch_labels = batch_labels.to(device=device_train).float()

                optimizer.zero_grad()
                if classification:
                    batch_labels = (batch_labels > config_dict['threshold'])
                    batch_labels = batch_labels.float()

                output, h1, _ = net(batch_samples_input1, h1_zero)

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
                    loss = sr.update_and_get_weighted_loss(batch_labels=batch_labels, unweighted_loss=loss)
                    error = torch.isnan(loss).any().item() or torch.isinf(loss).any().item()
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
                    output = (output > config_dict['threshold'])
                    batch_labels = (batch_labels > config_dict['threshold'])
                else:
                    output = (output >= 0.5)
                accuracy_train += (output == batch_labels).float().mean()
                n += 1
            _t_stop = time.time()
            logging.debug(f"Training time for 1 epoch : {_t_stop - _t_start} s")
            accuracy_train /= n
            loss_train /= n

            _t_start = time.time()
        output_validation, labels_validation, loss_validation, accuracy_validation, tp, tn, fp, fn = run_inference(validation_loader, criterion, net,
                                                                                                                   device_val, hidden_size,
                                                                                                                   nb_rnn_layers, classification,
                                                                                                                   batch_size_validation, config_dict['threshold'])
        f1_validation, precision_validation, recall_validation = get_metrics(tp, fp, fn)

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
                }, config_dict['path_dataset'] / experiment_name, _use_new_zipfile_serialization=False)
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
                }, config_dict['path_dataset'] / (experiment_name + "_on_loss"), _use_new_zipfile_serialization=False)
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


def get_trained_model(config_dict, path_experiments):
    experiment_name = config_dict['experiment_name']
    device_inference = config_dict["device_inference"]
    classification = config_dict["classification"]
    if device_inference.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"
    net = PortiloopNetwork(config_dict).to(device=device_inference)
    file_exp = experiment_name
    file_exp += "" if classification else "_on_loss"
    path_experiments = Path(path_experiments)
    if not device_inference.startswith("cuda"):
        checkpoint = torch.load(path_experiments / file_exp, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_experiments / file_exp)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net


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
        checkpoint = torch.load(path_experiments / file_exp, map_location=torch.device('cpu'))
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


def get_final_model_config_dict(index=0, split_i=0):
    """
    Configuration dictionary of the final 1-input pre-trained model presented in the Portiloop paper.

    Args:
        index: last number in the name of the pre-trained model (several are provided)
        split_i: index of the random train/validation/test split (you can ignore this for inference)

    Returns:
        configuration dictionary of the pre-trained model
    """
    c_dict = {'experiment_name': f'sanity_check_final_model_3',
              'device_train': 'cuda',
              'device_val': 'cuda',
              'device_inference': 'cpu',
              'nb_epoch_max': 150,
              'max_duration': 257400,
              'nb_epoch_early_stopping_stop': 100,
              'early_stopping_smoothing_factor': 0.1,
              'fe': 250,
              'nb_batch_per_epoch': 1000,
              'first_layer_dropout': False,
              'power_features_input': False,
              'dropout': 0.5,
              'adam_w': 0.01,
              'distribution_mode': 0,
              'classification': True,
              'reg_balancing': 'none',
              'split_idx': split_i,
              'validation_network_stride': 1,
              'nb_conv_layers': 3,
              'seq_len': 50,
              'nb_channel': 31,
              'hidden_size': 7,
              'seq_stride_s': 0.170,
              'nb_rnn_layers': 1,
              'RNN': True,
              'envelope_input': False,
              'lr_adam': 0.0005,
              'batch_size': 256,
              'stride_pool': 1,
              'stride_conv': 1,
              'kernel_conv': 7,
              'kernel_pool': 7,
              'dilation_conv': 1,
              'dilation_pool': 1,
              'nb_out': 18,
              'time_in_past': 8.5,
              'estimator_size_memory': 188006400}
    return c_dict

def get_configs(exp_name, test_set, seed_exp):
    """
    Get the configuration dictionaries containgin information about:
        - Paths where data is stored
        - Model info
        - Data info
    """

    config = {
        # Path info
        'path_dataset': Path(__file__).absolute().parent.parent.parent / 'dataset',
        'filename_regression_dataset': f"dataset_regression_{PHASE}_big_250_matlab_standardized_envelope_pf.txt",
        'filename_classification_dataset': f"dataset_classification_{PHASE}_big_250_matlab_standardized_envelope_pf.txt",
        'subject_list': f"subject_sequence_{PHASE}_big.txt",
        'subject_list_p1': f"subject_sequence_p1_big.txt",
        'subject_list_p2': f"subject_sequence_p2_big.txt",

        # Experiment info
        'experiment_name': exp_name,
        'seed_exp': seed_exp,
        'test_set': test_set,

        # Training hyperparameters
        'batch_size': 256,
        'dropout': 0.5,
        'adam_w': 0.01,
        'reg_balancing': 'none',
        'lr_adam': 0.0005,

        # Stopping parameters
        'nb_epoch_max': 150,
        'max_duration': 257400,
        'nb_epoch_early_stopping_stop': 100,
        'early_stopping_smoothing_factor': 0.1,

        # Model info
        'first_layer_dropout': False,
        'power_features_input': False,
        'RNN': True,
        'envelope_input': False,
        'classification': True,

        # CNN stuff
        'nb_conv_layers': 3,
        'nb_channel': 31,
        'hidden_size': 7,
        'stride_pool': 1,
        'stride_conv': 1,
        'kernel_conv': 7,
        'kernel_pool': 7,
        'dilation_conv': 1,
        'dilation_pool': 1,

        # RNN stuff
        'nb_rnn_layers': 1,
        'nb_out': 18,

        # Attention stuff
        'max_h_length': 50, # How many time steps to consider in the attention

        # IDK
        'time_in_past': 8.5,
        'estimator_size_memory': 188006400,

        # Device info
        'device_train': 'cuda',
        'device_val': 'cuda',
        'device_inference': 'cpu',

        # Data info
        'fe': 250,
        'validation_network_stride': 1,
        'phase': PHASE,
        'split_idx': 0,
        'threshold': 0.5,
        'window_size': 54,
        'seq_stride': 42,
        'nb_batch_per_epoch': 1000,
        'distribution_mode': 0,
        'seq_len': 50,
        'seq_stride_s': 0.170,
        'window_size_s': 0.218,
        'len_segment_s': LEN_SEGMENT,

    }

    return config



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--output_file', type=str, default=None)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--test_set', dest='test_set', action='store_true')
    feature_parser.add_argument('--no_test_set', dest='test_set', action='store_false')
    parser.set_defaults(test_set=True)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s', filename=args.output_file, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # Get configuration
    exp_name = args.experiment_name
    test_set = args.test_set
    exp_seed = seed()
    config_dict = get_configs(exp_name, test_set, seed)

    # Run experiment
    WANDB_PROJECT_RUN = f"{PHASE}-dataset-public"

    run(config_dict=config_dict, wandb_project=WANDB_PROJECT_RUN, save_model=True, unique_name=False)
else:
    ABLATION = 0
    PHASE = 'full'
    TEST_SET = True

    # threshold_list = {'p1': 0.2, 'p2': 0.35, 'full': 0.5}  # full = p1 + p2
    # THRESHOLD = threshold_list[PHASE]
    # WANDB_PROJECT_RUN = f"tests_yann"

    filename_regression_dataset = f"dataset_regression_{PHASE}_big_250_matlab_standardized_envelope_pf.txt"
    filename_classification_dataset = f"dataset_classification_{PHASE}_big_250_matlab_standardized_envelope_pf.txt"
    subject_list = f"subject_sequence_{PHASE}_big.txt"
    subject_list_p1 = f"subject_sequence_p1_big.txt"
    subject_list_p2 = f"subject_sequence_p2_big.txt"
