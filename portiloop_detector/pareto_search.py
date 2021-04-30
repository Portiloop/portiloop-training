"""
Pareto-optimal hyperparameter search (meta-learning)
"""
from portiloop_detector_training import PortiloopNetwork, SignalDataset, get_class_idxs, ValidationSampler, RandomSampler, out_dim, get_accuracy_and_loss_pytorch

# all imports
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle as pkl
from pathlib import Path
from torch.utils.data import DataLoader
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
import copy
import wandb
import datetime

# all constants (no hyperparameters here!)

THRESHOLD = 0.5
WANDB_PROJECT = "pareto"

filename_dataset = "dataset_big_fusion_standardized_envelope_pf.txt"

path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
path_pareto = Path(__file__).absolute().parent.parent / 'pareto'

# path = "/content/drive/MyDrive/Data/MASS/"
# path_dataset = Path(path)
# path_pareto = Path("/content/drive/MyDrive/Data/pareto_results/")

div_val_samp = 32

MAX_META_ITERATIONS = 1000  # maximum number of experiments
EPOCHS_PER_EXPERIMENT = 1  # experiments are evaluated after this number of epoch by the meta learner

EPSILON_NOISE = 0.1  # a completely random model will be selected this portion of the time, otherwise, it is sampled from a gaussian
EPSILON_EXP_NOISE = 0.05  # a random experiment is selected within all sampled experiments this portion of the time

MAX_NB_PARAMETERS = 100000  # everything over this number of parameters will be discarded
MAX_LOSS = 0.1  # to normalize distances

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

NB_BATCH_PER_EPOCH = 10000

RUN_NAME = "pareto_search_5"

NB_SAMPLED_MODELS_PER_ITERATION = 500  # number of models sampled per iteration, only the best predicted one is selected

DEFAULT_META_EPOCHS = 100  # default number of meta-epochs before entering meta train/val training regime
START_META_TRAIN_VAL_AFTER = 200  # minimum number of experiments in the dataset before using a validation set
META_TRAIN_VAL_RATIO = 0.8  # portion of experiments in meta training sets
MAX_META_EPOCHS = 1000  # surrogate training will stop after this number of meta-training epochs if the model doesn't converge
META_EARLY_STOPPING = 10  # meta early stopping after this number of unsuccessful meta epochs


# run:

def run(config_dict):
    _t_start = time.time()
    nb_epoch_max = config_dict["nb_epoch_max"]
    nb_batch_per_epoch = config_dict["nb_batch_per_epoch"]
    batch_size = config_dict["batch_size"]
    seq_len = config_dict["seq_len"]
    window_size_s = config_dict["window_size_s"]
    fe = config_dict["fe"]
    seq_stride_s = config_dict["seq_stride_s"]
    lr_adam = config_dict["lr_adam"]
    hidden_size = config_dict["hidden_size"]
    device_val = config_dict["device_val"]
    device_train = config_dict["device_train"]
    nb_rnn_layers = config_dict["nb_rnn_layers"]
    adam_w = config_dict["adam_w"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    net = PortiloopNetwork(config_dict).to(device=device_train)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr_adam, weight_decay=adam_w)

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

    best_model_loss_validation = 1

    h1_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    h2_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    for epoch in range(nb_epoch_max):

        # print(f"DEBUG: epoch: {epoch}")

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
            loss.backward()
            optimizer.step()

        _, loss_validation, _, _, _ = get_accuracy_and_loss_pytorch(
            validation_loader, criterion, net, device_val, hidden_size, nb_rnn_layers)
        if loss_validation < best_model_loss_validation:
            best_model_loss_validation = loss_validation
    return best_model_loss_validation


# hyperparameters

# batch_size_range_t = ["i", 256, 256]
# lr_adam_range_t = ["f", 0.0003, 0.0003]

seq_len_range_t = ["i", 10, 50]
kernel_conv_range_t = ["i", 3, 9]
kernel_pool_range_t = ["i", 3, 5]
stride_conv_range_t = ["i", 1, 5]
stride_pool_range_t = ["i", 1, 5]
dilation_conv_range_t = ["i", 1, 5]
dilation_pool_range_t = ["i", 1, 5]
nb_channel_range_t = ["i", 1, 50]
hidden_size_range_t = ["i", 2, 100]
window_size_s_range_t = ["f", 0.05, 0.1]
seq_stride_s_range_t = ["f", 0.05, 0.1]
nb_conv_layers_range_t = ["i", 1, 7]
nb_rnn_layers_range_t = ["i", 1, 5]


# dropout_range_t = ["f", 0.5, 0.5]
# first_layer_dropout_range_t = ["b", False, False]
# power_features_input_range_t = ["b", False, False]
# adam_w_range_t = ["f", 0.01, 0.01]


def clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


def sample_from_range(range_t, gaussian_mean=None, gaussian_std_factor=0.1):
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


def same_config_dict(config1, config2):
    flag = 0
    if config1["seq_len"] != config2["seq_len"]:
        flag += 1
    if config1["nb_channel"] != config2["nb_channel"]:
        flag += 1
    #  config_dict["dropout"], unrounded["dropout"] = sample_from_range(dropout_range_t)
    if config1["hidden_size"] != config2["hidden_size"]:
        flag += 1
    if config1["seq_stride_s"] != config2["seq_stride_s"]:
        flag += 1
    # config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t)
    if config1["nb_rnn_layers"] != config2["nb_rnn_layers"]:
        flag += 1
    #    config_dict["adam_w"], unrounded["adam_w"] = sample_from_range(adam_w_range_t)
    if config1["window_size_s"] != config2["window_size_s"]:
        flag += 1
    if config1["nb_conv_layers"] != config2["nb_conv_layers"]:
        flag += 1
    if config1["stride_pool"] != config2["stride_pool"]:
        flag += 1
    if config1["stride_conv"] != config2["stride_conv"]:
        flag += 1
    if config1["kernel_conv"] != config2["kernel_conv"]:
        flag += 1
    if config1["kernel_pool"] != config2["kernel_pool"]:
        flag += 1
    if config1["dilation_conv"] != config2["dilation_conv"]:
        flag += 1
    if config1["dilation_pool"] != config2["dilation_pool"]:
        flag += 1
    return flag == 0


def sample_config_dict(name, previous_exp, all_exp):
    config_dict = dict(experiment_name=name,
                       device_train="cuda:0",
                       device_val="cpu",
                       nb_epoch_max=EPOCHS_PER_EXPERIMENT,
                       max_duration=int(71.5 * 3600),
                       nb_epoch_early_stopping_stop=200,
                       early_stopping_smoothing_factor=0.01,
                       fe=250,
                       nb_batch_per_epoch=NB_BATCH_PER_EPOCH)

    noise = random.choices(population=[True, False], weights=[EPSILON_NOISE, 1.0 - EPSILON_NOISE])[0]

    unrounded = {}

    # constant things:

    config_dict["RNN"] = True
    config_dict["envelope_input"] = True
    config_dict["batch_size"] = 256
    config_dict["first_layer_dropout"] = False
    config_dict["power_features_input"] = False
    config_dict["dropout"] = 0.5
    config_dict["lr_adam"] = 0.0003
    config_dict["adam_w"] = 0.01
    flag_in_exps = True
    while flag_in_exps:
        nb_out = 0
        while nb_out < 1:

            if previous_exp == {} or noise:
                # sample completely randomly
                config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t)
                config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t)
                #  config_dict["dropout"], unrounded["dropout"] = sample_from_range(dropout_range_t)
                config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t)
                config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t)
                # config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t)
                config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t)
                #    config_dict["adam_w"], unrounded["adam_w"] = sample_from_range(adam_w_range_t)
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
                # previous_experiment = random.choice(pareto_front)
                previous_unrounded = previous_exp["unrounded"]
                config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t, previous_unrounded["seq_len"])
                config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t, previous_unrounded["nb_channel"])
                # config_dict["dropout"], unrounded["dropout"] = sample_from_range(dropout_range_t, previous_unrounded["dropout"])
                config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t, previous_unrounded["hidden_size"])
                config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t, previous_unrounded["seq_stride_s"])
                # config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t, previous_unrounded["lr_adam"])
                config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t, previous_unrounded["nb_rnn_layers"])
                # config_dict["adam_w"], unrounded["adam_w"] = sample_from_range(adam_w_range_t, previous_unrounded["adam_w"])
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
        flag_in_exps = False
        for exp in all_experiments:
            if same_config_dict(exp['config_dict'], config_dict):
                flag_in_exps = True
                print(f"DEBUG : config already tried = {config_dict}")
                break

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

        self.fc1 = nn.Linear(in_features=13,  # nb hyperparameters
                             out_features=13*25)  # in SMBO paper : 25 * hyperparameters... Seems huge

        self.d1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=13*25,
                             out_features=13*25)

        self.d2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=13*25,
                             out_features=1)

    def to(self, device):
        super(SurrogateModel, self).to(device)
        self.device = device

    def forward(self, config_dict):
        x_list = [config_dict["seq_len"],
                  config_dict["nb_channel"],
                  # config_dict["dropout"],
                  config_dict["hidden_size"],
                  config_dict["seq_stride_s"],
                  # config_dict["lr_adam"],
                  config_dict["nb_rnn_layers"],
                  # config_dict["adam_w"],
                  config_dict["window_size_s"],
                  config_dict["nb_conv_layers"],
                  config_dict["stride_pool"],
                  config_dict["stride_conv"],
                  config_dict["kernel_conv"],
                  config_dict["kernel_pool"],
                  config_dict["dilation_conv"],
                  config_dict["dilation_pool"]]

        x_tensor = torch.tensor(x_list).to(self.device)

        x_tensor = F.relu(self.d1(self.fc1(x_tensor)))
        x_tensor = F.relu(self.d2(self.fc2(x_tensor)))

        # x_tensor = F.relu(self.fc1(x_tensor))
        # x_tensor = F.relu(self.fc2(x_tensor))

        x_tensor = self.fc3(x_tensor)

        return x_tensor


def order_exp(exp):
    return exp["cost_software"]


def sort_pareto(pareto_front):
    pareto_front.sort(key=order_exp)
    return pareto_front


def update_pareto(experiment, pareto):
    to_remove = []
    if len(pareto) == 0:
        dominates = True
    else:
        dominates = True
        for i, ep in enumerate(pareto):
            if ep["cost_software"] <= experiment["cost_software"] and ep["cost_hardware"] <= experiment["cost_hardware"]:
                dominates = False
            if ep["cost_software"] > experiment["cost_software"] and ep["cost_hardware"] > experiment["cost_hardware"]:  # remove ep from pareto
                to_remove.append(i)
    to_remove.sort(reverse=True)
    for i in to_remove:
        pareto.pop(i)
    if dominates:
        pareto.append(experiment)
    pareto = sort_pareto(pareto)
    return pareto


def dominates_pareto(experiment, pareto):
    if len(pareto) == 0:
        dominates = True
    else:
        dominates = True
        for ep in pareto:
            if ep["cost_software"] <= experiment["cost_software"] and ep["cost_hardware"] <= experiment["cost_hardware"]:
                dominates = False
    return dominates


def train_surrogate(net, all_experiments):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0.01, nesterov=False)
    loss = nn.MSELoss()
    if len(all_experiments) < START_META_TRAIN_VAL_AFTER:  # no train/val
        net.train()
        losses = []
        nb_epochs = min(len(all_experiments), 100)
        for epoch in range(nb_epochs):
            random.shuffle(all_experiments)
            samples = [exp["config_dict"] for exp in all_experiments]
            labels = [exp["cost_software"] for exp in all_experiments]
            for i, sample in enumerate(samples):
                optimizer.zero_grad()
                pred = net(sample)
                targ = torch.tensor([labels[i], ])
                assert pred.shape == targ.shape, f"pred.shape:{pred.shape} != targ.shape:{targ.shape}"
                l = loss(pred, targ)
                losses.append(l.item())
                l.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
        mean_loss = np.mean(losses)
    else:  # train/val regime
        random.shuffle(all_experiments)
        sep = int(META_TRAIN_VAL_RATIO * len(all_experiments))
        training_set = all_experiments[:sep]
        validation_set = all_experiments[sep:]
        best_val_loss = np.inf
        best_model = None
        early_stopping_counter = 0
        for epoch in range(MAX_META_EPOCHS):
            # training
            net.train()
            random.shuffle(training_set)
            samples = [exp["config_dict"] for exp in training_set]
            labels = [exp["cost_software"] for exp in training_set]
            for i, sample in enumerate(samples):
                optimizer.zero_grad()
                pred = net(sample)
                targ = torch.tensor([labels[i], ])
                assert pred.shape == targ.shape, f"pred.shape:{pred.shape} != targ.shape:{targ.shape}"
                l = loss(pred, targ)
                l.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
            # validation
            net.eval()
            samples = [exp["config_dict"] for exp in validation_set]
            labels = [exp["cost_software"] for exp in validation_set]
            losses = []
            with torch.no_grad():
                for i, sample in enumerate(samples):
                    pred = net(sample)
                    targ = torch.tensor([labels[i], ])
                    assert pred.shape == targ.shape, f"pred.shape:{pred.shape} != targ.shape:{targ.shape}"
                    l = loss(pred, targ)
                    losses.append(l.item())
            mean_loss = np.mean(losses)
            if mean_loss < best_val_loss:
                best_val_loss = mean_loss
                early_stopping_counter = 0
                best_model = deepcopy(net)
            else:
                early_stopping_counter += 1
            # early stopping:
            if early_stopping_counter >= META_EARLY_STOPPING:
                print(f"DEBUG: meta training converged at epoch:{epoch} (-{META_EARLY_STOPPING})")
                break
            elif epoch == MAX_META_EPOCHS - 1:
                print(f"DEBUG: meta training did not converge after epoch:{epoch}")
                break
        net = best_model
        mean_loss = best_val_loss
    net.eval()
    return net, mean_loss


def wandb_plot_pareto(all_experiments, ordered_pareto_front):
    plt.clf()
    # all experiments minus last dot:
    x_axis = [exp["cost_hardware"] for exp in all_experiments[:-1]]
    y_axis = [exp["cost_software"] for exp in all_experiments[:-1]]
    plt.plot(x_axis, y_axis, 'bo')
    # pareto:
    x_axis = [exp["cost_hardware"] for exp in ordered_pareto_front]
    y_axis = [exp["cost_software"] for exp in ordered_pareto_front]
    plt.plot(x_axis, y_axis, 'ro-')
    # last dot
    plt.plot(all_experiments[-1]["cost_hardware"], all_experiments[-1]["cost_software"], 'go')

    plt.xlabel(f"nb parameters")
    plt.ylabel(f"validation loss")
    plt.ylim(top=0.1)
    plt.draw()
    return wandb.Image(plt)


# Custom Pareto efficiency (distance from Pareto)

def dist_p_to_ab(v_a, v_b, v_p):
    l2 = np.linalg.norm(v_a - v_b) ** 2
    if l2 == 0.0:
        return np.linalg.norm(v_p - v_a)
    t = max(0.0, min(1.0, np.dot(v_p - v_a, v_b - v_a) / l2))
    projection = v_a + t * (v_b - v_a)
    return np.linalg.norm(v_p - projection)


def vector_exp(experiment):
    return np.array([experiment["cost_software"] / MAX_LOSS, experiment["cost_hardware"] / MAX_NB_PARAMETERS])


def pareto_efficiency(experiment, all_experiments):
    if len(all_experiments) < 1:
        return 0.0

    nb_dominating = 0
    nb_dominated = 0
    for exp in all_experiments:
        if exp["cost_software"] < experiment["cost_software"] and exp["cost_hardware"] < experiment["cost_hardware"]:
            nb_dominating += 1
        if exp["cost_software"] > experiment["cost_software"] and exp["cost_hardware"] > experiment["cost_hardware"]:
            nb_dominated += 1

    score_not_dominated = 1.0 - float(nb_dominating) / len(all_experiments)
    score_dominating = nb_dominated / len(all_experiments)
    return score_dominating + score_not_dominated

    # v_p = vector_exp(experiment)
    # dominates = True
    # all_dists = []
    # for i in range(len(pareto_front)):
    #     exp = pareto_front[i]
    #     if exp["cost_software"] <= experiment["cost_software"] and exp["cost_hardware"] <= experiment["cost_hardware"]:
    #         dominates = False
    #     if i < len(pareto_front) - 1:
    #         next = pareto_front[i + 1]
    #         v_a = vector_exp(exp)
    #         v_b = vector_exp(next)
    #         dist = dist_p_to_ab(v_a, v_b, v_p)
    #         all_dists.append(dist)
    # assert len(all_dists) >= 1
    # res = min(all_dists)  # distance to pareto
    # if not dominates:
    #     res *= -1.0
    # # subtract density around number of parameters
    # return res


def exp_max_pareto_efficiency(experiments, pareto_front, all_experiments):
    assert len(experiments) >= 1
    noise = random.choices(population=[True, False], weights=[EPSILON_EXP_NOISE, 1.0 - EPSILON_EXP_NOISE])[0]
    if noise or len(pareto_front) == 0:
        return random.choice(experiments)
    else:
        assert len(all_experiments) != 0
        histo = np.histogram([exp["cost_hardware"] for exp in all_experiments], bins=100, density=True, range=(0, MAX_NB_PARAMETERS))

        max_efficiency = -np.inf
        best_exp = None
        for exp in experiments:
            efficiency = pareto_efficiency(exp, all_experiments)
            assert histo[1][0] <= exp["cost_hardware"] <= histo[1][-1]
            idx = np.where(histo[1] <= exp["cost_hardware"])[0][-1]
            nerf = histo[0][idx] * MAX_NB_PARAMETERS
            efficiency -= nerf
            if efficiency >= max_efficiency:
                max_efficiency = efficiency
                best_exp = exp
                best_efficiency = efficiency + nerf
                best_nerf = nerf
        assert best_exp is not None
        print(f"DEBUG: selected {best_exp['cost_hardware']}: efficiency:{best_efficiency}, nerf:{best_nerf}")
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
















# Networking ===========================================================================

import socket
from threading import Lock, Thread
from requests import get
import pickle as pkl
import os


WAIT_BEFORE_RECONNECTION = 500.0

SOCKET_TIMEOUT_COMMUNICATE = 500.0

SOCKET_TIMEOUT_ACCEPT_META = 500.0
SOCKET_TIMEOUT_ACCEPT_WORKER = 500.0

ACK_TIMEOUT_SERVER_TO_WORKER = 500.0
ACK_TIMEOUT_SERVER_TO_META = 500.0
ACK_TIMEOUT_META_TO_SERVER = 500.0
ACK_TIMEOUT_WORKER_TO_SERVER = 500.0

LOOP_SLEEP_TIME = 1.0

PORT_META = 66666
PORT_WORKER = 66667

LEN_QUEUE_TO_LAUNCH = 5


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    print(sx + str(s))


def send_ack(sock):
    return send_object(sock, None, ack=True)


def send_object(sock, obj, ack=False):
    """
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ack:
        msg = bytes(f"{'ACK':<{cfg.HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{cfg.HEADER_SIZE}}", 'utf-8') + msg
        if cfg.PRINT_BYTESIZES:
            print_with_timestamp(f"Sending {len(msg)} bytes.")
    try:
        sock.sendall(msg)
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != cfg.HEADER_SIZE:
        try:
            recv_msg = sock.recv(cfg.HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print_with_timestamp(f"DEBUG: l:{l}")
    # print_with_timestamp("DEBUG: data len:", msg[:HEADER_SIZE])
    # print_with_timestamp(f"DEBUG: msg[:4]: {msg[:4]}")
    if msg[:4] == b'PING' or msg[:4] == b'PONG':
        if msg[:4] == b'PING':
            send_pong(sock)
        return 'PINGPONG'
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:cfg.HEADER_SIZE])
    # print_with_timestamp(f"DEBUG: receiving {msglen} bytes")
    t_start = time.time()
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(cfg.BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print_with_timestamp(f"DEBUG2: l:{l}")
    # print_with_timestamp("DEBUG: final data len:", l)
    # print_with_timestamp(f"DEBUG: finished receiving after {time.time() - t_start}s.")
    send_ack(sock)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # to reuse address on Linux
    s.bind((ip_bind, port_bind))
    s.listen(5)
    return s


def get_connected_socket(timeout, ip_connect, port_connect):
    """
    returns the connected socket
    returns None if connect failed
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip_connect, port_connect))
    except OSError:  # connection broken or timeout
        print_with_timestamp(f"INFO: connect() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        # print_with_timestamp(f"INFO: accept() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        if conn is not None:
            conn.close()
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    print_with_timestamp(f"DEBUG: start select")
    _, wl, xl = select.select([], [conn], [conn], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    print_with_timestamp(f"DEBUG: end select")
    if len(xl) != 0:
        print_with_timestamp("INFO: error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print_with_timestamp("INFO: outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print_with_timestamp("INFO: send_object() failed, closing socket")
        conn.close()
        return False
    return True


def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print_with_timestamp("INFO: error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print_with_timestamp("INFO: error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        # print_with_timestamp(f"DEBUG: received obj:{obj}")
        return True, obj





class Server:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    If trainer_on_localhost is True, the server only listens on trainer_on_localhost. Then the trainer is expected to talk on trainer_on_localhost.
    Otherwise, the server also listens to the local ip and the trainer is expected to talk on the local ip (port forwarding).
    """
    def __init__(self, samples_per_server_packet=1000):
        self.__finished_lock = Lock()
        self.__finished = []
        self.__to_launch_lock = Lock()
        self.__to_launch = []
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print_with_timestamp(f"INFO REDIS: local IP: {self.local_ip}")
        print_with_timestamp(f"INFO REDIS: public IP: {self.public_ip}")

        Thread(target=self.__workers_thread, args=('', ), kwargs={}, daemon=True).start()
        Thread(target=self.__metas_thread, args=('', ), kwargs={}, daemon=True).start()

    def __metas_thread(self, ip):
        """
        This waits for new potential Trainers to connect
        When a new Trainer connects, this instantiates a new thread to handle it
        """
        while True:  # main redis loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_META, ip, PORT_META)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                # print_with_timestamp("DEBUG: accept_or_close_socket failed in trainers thread")
                continue
            print_with_timestamp(f"INFO METAS THREAD: server connected by meta at address {addr}")
            Thread(target=self.__meta_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __meta_thread(self, conn):
        """
        This periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        is_working = False
        while True:
            # send samples
            if not is_working:
                self.__to_launch_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__to_launch) < LEN_QUEUE_TO_LAUNCH:  # send request to meta
                    if not wait_ack:
                        self.__finished_lock.acquire()
                        obj = deepcopy(self.__finished)
                        self.__finished = []
                        self.__finished_lock.release()
                        if select_and_send_or_close_socket(obj, conn):
                            is_working = True
                            wait_ack = True
                            ack_time = time.time()
                        else:
                            print_with_timestamp("INFO: failed sending object to meta")
                            self.__to_launch_lock.release()
                            break
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_SERVER_TO_META:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                self.__to_launch_lock.release()  # END BUFFER LOCK.........................................................
            # checks for weights
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("DEBUG: poll failed in meta thread")
                break
            elif obj is not None and obj != 'ACK':
                is_working = False
                print_with_timestamp(f"DEBUG INFO: meta thread received obj")
                self.__to_launch_lock.acquire()  # LOCK.......................................................
                self.__to_launch.append(obj)
                self.__to_launch_lock.release()  # END LOCK...................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt

    def __workers_thread(self, ip):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_WORKER, ip, PORT_WORKER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO WORKERS THREAD: server connected by worker at address {addr}")
            Thread(target=self.__worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        ack_time = time.time()
        wait_ack = False
        is_working = False
        while True:
            # send weights
            if not is_working:
                self.__to_launch_lock.acquire()  # LOCK...............................................................
                if len(self.__to_launch) > 0:  # exps to be sent
                    if not wait_ack:
                        obj = self.__to_launch.pop(0)
                        if select_and_send_or_close_socket(obj, conn):
                            is_working = True
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__to_launch_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in worker thread")
                            break
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"INFO: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_SERVER_TO_WORKER:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__to_launch_lock.release()
                            break
                self.__to_launch_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("DEBUG: poll failed in worker thread")
                break
            elif obj is not None and obj != 'ACK':
                is_working = False
                print_with_timestamp(f"DEBUG INFO: worker thread received obj")
                self.__finished_lock.acquire()  # BUFFER LOCK.............................................................
                self.__finished.append(obj)
                self.__finished_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(LOOP_SLEEP_TIME)


# TRAINER: ==========================================


class MetaInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self, server_ip=None):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'
        self.recv_tiemout = cfg.RECV_TIMEOUT_TRAINER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_TRAINER, self.redis_ip, cfg.PORT_TRAINER)
            if s is None:
                print_with_timestamp("DEBUG: get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    if not wait_ack:
                        obj = self.__weights
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__weights_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_TRAINER_TO_REDIS:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            wait_ack = False
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp("DEBUG: poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print_with_timestamp(f"DEBUG INFO: trainer interface received obj")
                    recv_time = time.time()
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_tiemout:
                    print_with_timestamp(f"DEBUG: Timeout in TrainerInterface, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule (sac_models.py)
        broadcasts the model's weights to all connected RolloutWorkers
        """
        t0 = time.time()
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        t1 = time.time()
        torch.save(model.state_dict(), self.model_path)
        t2 = time.time()
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        t3 = time.time()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
        print_with_timestamp(f"DEBUG: broadcast_model: lock acquire: {t1 - t0}s, save dict: {t2 - t1}s, read dict: {t3 - t2}s")

    def retrieve_buffer(self):
        """
        returns a copy of the TrainerInterface's local buffer, and clears it
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        buffer_copy = deepcopy(self.__buffer)
        self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        return buffer_copy


# ROLLOUT WORKER: ===================================


class RolloutWorker:
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            get_local_buffer_sample: callable,
            device="cpu",
            redis_ip=None,
            samples_per_worker_packet=1000,  # The RolloutWorker waits for this number of samples before sending
            max_samples_per_episode=1000000,  # If the episode is longer than this, it is reset by the RolloutWorker
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,  # if 0, doesn't save model history, else, the model is saved every model_history episode
    ):
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = get_local_buffer_sample
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.actor = actor_module_cls(obs_space, act_space).to(device)
        self.device = device
        if os.path.isfile(self.model_path):
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = samples_per_worker_packet
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'
        self.recv_timeout = cfg.RECV_TIMEOUT_WORKER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.redis_ip, cfg.PORT_ROLLOUT)
            if s is None:
                print_with_timestamp("DEBUG: get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    print_with_timestamp("DEBUG: new batch available")
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print_with_timestamp("DEBUG: select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_WORKER_TO_REDIS:
                            print_with_timestamp("INFO: ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            wait_ack = False
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp(f"INFO: rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print_with_timestamp(f"DEBUG INFO: rollout worker received obj")
                    recv_time = time.time()
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    print_with_timestamp(f"DEBUG: Timeout in RolloutWorker, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, deterministic=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=deterministic)
            # action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        # action, = partition(action)
        return action

    def reset(self, collect_samples):
        obs = None
        act = self.env.default_action.astype(np.float32)
        new_obs = self.env.reset()
        rew = 0.0
        done = False
        info = {}
        if collect_samples:
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, done)
            sample = self.get_local_buffer_sample(act, new_obs, rew, done, info)
            self.buffer.append_sample(sample)
        return new_obs

    def step(self, obs, deterministic, collect_samples, last_step=False):
        act = self.act(obs, deterministic=deterministic)
        new_obs, rew, done, info = self.env.step(act)
        if collect_samples:
            stored_done = done
            if last_step and not done:  # ignore done when stopped by step limit
                info["__no_done"] = True
            if "__no_done" in info:
                stored_done = False
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, stored_done)
            sample = self.get_local_buffer_sample(act, new_obs, rew, stored_done, info)
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, done, info

    def collect_train_episode(self, max_samples):
        """
        collects a maximum of n training transitions (from reset to done)
        stores episode and train return in the local buffer of the worker
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=True)
        for i in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, deterministic=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples, train=False):
        """
        collects a maximum of n test transitions (from reset to done)
        stores test return in the local buffer of the worker
        """
        while True:
            ret = 0.0
            steps = 0
            obs = self.reset(collect_samples=False)
            for _ in range(max_samples):
                obs, rew, done, info = self.step(obs=obs, deterministic=not train, collect_samples=False)
                ret += rew
                steps += 1
                if done:
                    break
            self.buffer.stat_test_return = ret
            self.buffer.stat_test_steps = steps

    def run_test_episode(self, max_samples):
        """
        collects a maximum of n test transitions (from reset to done)
        stores test return in the local buffer of the worker
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=False)
        for _ in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, deterministic=True, collect_samples=False)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=20):  # TODO: check number of collected samples are collected before sending
        episode = 0
        while True:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("INFO: running test episode")
                self.run_test_episode(self.max_samples_per_episode)
            print_with_timestamp("INFO: collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("INFO: copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("INFO: checking for new weights")
            self.update_actor_weights()
            episode += 1
            # if self.crc_debug:
            #     break

    def profile_step(self, nb_steps=100):
        import torch.autograd.profiler as profiler
        obs = self.reset(collect_samples=True)
        use_cuda = True if self.device == 'cuda' else False
        print_with_timestamp(f"DEBUG: use_cuda:{use_cuda}")
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            obs = collate([obs], device=self.device)
            with profiler.record_function("pytorch_profiler"):
                with torch.no_grad():
                    action_distribution = self.actor(obs)
                    action = action_distribution.sample()
        print_with_timestamp(prof.key_averages().table(row_limit=20, sort_by="cpu_time_total"))

    def run_env_benchmark(self, nb_steps, deterministic=False):
        """
        This method is only compatible with rtgym environments
        """
        obs = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, done, info = self.step(obs=obs, deterministic=deterministic, collect_samples=False)
            if done:
                obs = self.reset(collect_samples=False)
        print_with_timestamp(f"Benchmark results:\n{self.env.benchmarks()}")

    def send_and_clear_buffer(self):
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer += self.buffer
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        updates the model with new weights from the trainer when available
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:
                f.write(self.__weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".pth", 'wb') as f:
                        f.write(self.__weights)
                    self._cur_hist_cpt = 0
                    print_with_timestamp("INFO: model weights saved in history")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print_with_timestamp("INFO: model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................













































# Main:

if __name__ == "__main__":

    logger = LoggerWandbPareto(RUN_NAME)

    all_experiments, pareto_front = load_files()

    if all_experiments is None:
        print(f"DEBUG: no meta dataset found, starting new run")
        all_experiments = []  # list of dictionaries
        pareto_front = []  # list of dictionaries, subset of all_experiments
        meta_model = SurrogateModel()
        meta_model.to(META_MODEL_DEVICE)
    else:
        print(f"DEBUG: existing meta dataset loaded")
        print("training new surrogate model...")
        meta_model = SurrogateModel()
        meta_model.to(META_MODEL_DEVICE)
        meta_model.train()
        meta_model, meta_loss = train_surrogate(meta_model, deepcopy(all_experiments))
        print(f"surrogate model loss: {meta_loss}")

    # main meta-learning procedure:

    for meta_iteration in range(MAX_META_ITERATIONS):
        num_experiment = len(all_experiments)
        print("---")
        print(f"ITERATION N° {meta_iteration}")

        exp = {}
        prev_exp = {}
        exps = []
        model_selected = False
        meta_model.eval()

        while not model_selected:
            exp = {}

            # sample model
            config_dict, unrounded = sample_config_dict(name=RUN_NAME + "_" + str(num_experiment), previous_exp=prev_exp, all_exp=all_experiments)

            nb_params = nb_parameters(config_dict)
            if nb_params > MAX_NB_PARAMETERS:
                continue

            with torch.no_grad():
                predicted_loss = meta_model(config_dict).item()

            exp["cost_hardware"] = nb_params
            exp["cost_software"] = predicted_loss
            exp["config_dict"] = config_dict
            exp["unrounded"] = unrounded

            exps.append(exp)

            if len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
                # select model
                model_selected = True
                exp = exp_max_pareto_efficiency(exps, pareto_front, all_experiments)

        config_dict = exp["config_dict"]
        predicted_loss = exp["cost_software"]
        nb_params = exp["cost_hardware"]

        print(f"config: {config_dict}")

        print(f"nb parameters: {nb_params}")
        print(f"predicted loss: {predicted_loss}")
        print("training...")

        exp["cost_software"] = run(config_dict)

        pareto_front = update_pareto(exp, pareto_front)
        all_experiments.append(exp)

        prev_exp = exp

        print(f"actual loss: {exp['cost_software']}")
        surprise = exp['cost_software'] - predicted_loss
        print(f"surprise: {surprise}")

        print("training new surrogate model...")

        meta_model = SurrogateModel()
        meta_model.to(META_MODEL_DEVICE)

        meta_model.train()
        meta_model, meta_loss = train_surrogate(meta_model, deepcopy(all_experiments))

        print(f"surrogate model loss: {meta_loss}")

        dump_files(all_experiments, pareto_front)
        logger.log(surrogate_loss=meta_loss, surprise=surprise, all_experiments=all_experiments, pareto_front=pareto_front)

    print(f"End of meta-training.")


