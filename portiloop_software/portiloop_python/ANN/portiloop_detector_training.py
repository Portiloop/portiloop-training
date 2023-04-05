"""
Main script for training an ANN.
"""

# all imports

import copy
import logging
import os
import random
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import wandb
from portiloop_software.portiloop_python.ANN.data.mass_data import get_dataloaders_mass
from portiloop_software.portiloop_python.ANN.data.moda_data import (
    generate_dataloader, generate_dataloader_unlabelled_offline)
from portiloop_software.portiloop_python.ANN.data.reg_balancing import (
    LabelDistributionSmoothing, SurpriseReweighting)
from portiloop_software.portiloop_python.ANN.models.lstm import PortiloopNetwork
from portiloop_software.portiloop_python.ANN.utils import LoggerWandb, get_configs, get_metrics, set_seeds


recall_validation_factor = 0.5
precision_validation_factor = 0.5

# all classes and functions:


def run_inference(dataloader, criterion, net, device, hidden_size, nb_rnn_layers, classification, batch_size_validation, threshold, recurrent=True):
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
    if recurrent:
        h1 = torch.zeros((nb_rnn_layers, batch_size_validation, hidden_size), device=device)

    # Run through the dataloader
    with torch.no_grad():
        # out_grus = []
        for batch_data in dataloader:
            # Get the current batch data
            batch_samples_input1, _, _, batch_labels = batch_data
            batch_samples_input1 = batch_samples_input1.to(device=device).float()
            batch_labels = batch_labels.to(device=device).float()
            if classification:
                batch_labels = (batch_labels > threshold)
                batch_labels = batch_labels.float()

            # Run the model
            # h1 = torch.zeros((nb_rnn_layers, batch_size_validation, hidden_size), device=device)
            if recurrent:
                output, h1, _ = net_copy(batch_samples_input1, h1)
            else:
                output = net_copy(batch_samples_input1)

            # out_grus.append(out_gru)
            # MAX_H_SIZE = 100
            # if len(out_grus) > MAX_H_SIZE:
            #     out_grus.pop(0)

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
    output_total = output_total.float()
    batch_labels_total = batch_labels_total.float()

    return output_total, batch_labels_total, loss


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


def train(train_loader, val_loader, model, recurrent, logger, save_model, unique_name, config_dict):
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

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    # Choose model type:
    net = copy.deepcopy(model)

    criterion =  nn.BCELoss(reduction='none')
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

    net = net.train()

    best_model_accuracy = 0
    best_epoch = 0
    best_model = None
    accuracy_train = None
    loss_train = None
    early_stopping_counter = 0
    loss_early_stopping = None
    h1_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)

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

                # Forward pass
                if recurrent:
                    output, _, _ = net(batch_samples_input1, h1_zero)
                else: 
                    output = net(batch_samples_input1)

                output = output.view(-1) # (output > THRESHOLD)

                # Get the labels for the batch
                batch_labels = (batch_labels > config_dict['threshold'])
                batch_labels = batch_labels.float()

                # Compute loss
                loss = criterion(output, batch_labels)
                loss = loss.mean()

                # Backward pass
                loss_train += loss.item()
                loss.backward()
                optimizer.step()

                # Compute accuracy
                output = (output >= 0.5)
                accuracy_train += (output == batch_labels).float().mean()
                n += 1
            _t_stop = time.time()
            logging.debug(f"Training time for 1 epoch : {_t_stop - _t_start} s")
            accuracy_train /= n
            loss_train /= n

            _t_start = time.time()

        output_validation, labels_validation, loss_validation = run_inference(
            val_loader, 
            criterion, 
            net, 
            device_val, 
            hidden_size,
            nb_rnn_layers, 
            classification,
            config_dict['batch_size_validation'], 
            config_dict['threshold'], 
            recurrent=recurrent)
        accuracy_validation, f1_validation, precision_validation, recall_validation = get_metrics(output_validation, labels_validation)

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
                }, config_dict['path_models'] / experiment_name, _use_new_zipfile_serialization=False)
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
                }, config_dict['path_models'] / (experiment_name + "_on_loss"), _use_new_zipfile_serialization=False)
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

        if early_stopping_counter > nb_epoch_early_stopping_stop:
            logging.debug("Early stopping.")
            break
    logging.debug("Delete logger")
    del logger
    logging.debug("Logger deleted")
    return best_model_loss_validation, best_model_f1_score_validation, best_epoch_early_stopping



def run(config_dict, wandb_project, save_model, unique_name, wandb_group):
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

    logger = LoggerWandb(experiment_name, config_dict, wandb_project, group=wandb_group)
    torch.seed()

    # Choose model type:
    net = PortiloopNetwork(config_dict).to(device=device_train)
    recurrent = True

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
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.debug("Use checkpoint model")

        first_epoch = checkpoint['epoch'] + 1
        recall_validation_factor = checkpoint['recall_validation_factor']
        precision_validation_factor = checkpoint['precision_validation_factor']
        best_model_on_loss_loss_validation = checkpoint['best_model_on_loss_loss_validation']
        best_model_f1_score_validation = checkpoint['best_model_f1_score_validation']
    except (ValueError, FileNotFoundError, RuntimeError):
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

    # Choose dataset to use
    train_loader, validation_loader, batch_size_validation, _, _, _ = generate_dataloader(config_dict)
    # batch_size_validation = 1
    config_dict["batch_size_validation"] = batch_size_validation

    train_loader_mass, validation_loader_mass = get_dataloaders_mass(config_dict)
    MASS = True
    if MASS:
        train_loader = train_loader_mass
        # validation_loader = validation_loader_mass

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

                output, _, _ = net(batch_samples_input1, h1_zero)

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


        output_validation, labels_validation, loss_validation = run_inference(validation_loader, criterion, net,
                                                                                                                   device_val, hidden_size,
                                                                                                                   nb_rnn_layers, classification,
                                                                                                                   batch_size_validation, config_dict['threshold'], recurrent=recurrent)
        accuracy_validation, f1_validation, precision_validation, recall_validation = get_metrics(output_validation, labels_validation)

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
                }, config_dict['path_models'] / experiment_name, _use_new_zipfile_serialization=False)
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
                }, config_dict['path_models'] / (experiment_name + "_on_loss"), _use_new_zipfile_serialization=False)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--experiment_group', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=-1)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--test_set', dest='test_set', action='store_true')
    feature_parser.add_argument('--no_test_set', dest='test_set', action='store_false')
    parser.set_defaults(test_set=True)

    # Parse arguments
    args = parser.parse_args()

    # Set seed
    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 1000000)

    # Set up logging
    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s', filename=args.output_file, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # Get configuration
    exp_name = args.experiment_name
    test_set = args.test_set
    set_seeds(seed)
    config_dict = get_configs(exp_name, test_set, seed)

    # Run experiment
    WANDB_PROJECT_RUN = f"full-dataset-public"

    run(
        config_dict=config_dict, 
        wandb_project=WANDB_PROJECT_RUN, 
        save_model=True, 
        unique_name=False, 
        wandb_group=args.experiment_group,
    )
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
