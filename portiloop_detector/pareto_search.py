"""
Pareto-optimal hyperparameter search (meta-learning)
"""
from portiloop_detector_training import PortiloopNetwork, SignalDataset, get_class_idxs, ValidationSampler, RandomSampler, out_dim, get_accuracy_and_loss_pytorch, filename_dataset, generate_dataloader

# all imports
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle as pkl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
import wandb

# all constants (no hyperparameters here!)

THRESHOLD = 0.5
WANDB_PROJECT = "pareto"

path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
path_pareto = Path(__file__).absolute().parent.parent / 'pareto'

# path = "/content/drive/MyDrive/Data/MASS/"
# path_dataset = Path(path)
# path_pareto = Path("/content/drive/MyDrive/Data/pareto_results/")

div_val_samp = 0

MAX_META_ITERATIONS = 1000  # maximum number of experiments
EPOCHS_PER_EXPERIMENT = 100  # experiments are evaluated after this number of epoch by the meta learner

EPSILON_NOISE = 0.25  # a completely random model will be selected this portion of the time, otherwise, it is sampled from a gaussian
EPSILON_EXP_NOISE = 0.1  # a random experiment is selected within all sampled experiments this portion of the time

MIN_NB_PARAMETERS = 1000  # everything below this number of parameters will be discarded
MAX_NB_PARAMETERS = 100000  # everything over this number of parameters will be discarded
MAX_LOSS = 0.1  # to normalize distances

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

NB_BATCH_PER_EPOCH = 10000

RUN_NAME = "pareto_search_9"

NB_SAMPLED_MODELS_PER_ITERATION = 500  # number of models sampled per iteration, only the best predicted one is selected

DEFAULT_META_EPOCHS = 100  # default number of meta-epochs before entering meta train/val training regime
START_META_TRAIN_VAL_AFTER = 200  # minimum number of experiments in the dataset before using a validation set
META_TRAIN_VAL_RATIO = 0.8  # portion of experiments in meta training sets
MAX_META_EPOCHS = 500  # surrogate training will stop after this number of meta-training epochs if the model doesn't converge
META_EARLY_STOPPING = 100  # meta early stopping after this number of unsuccessful meta epochs

NETWORK_EARLY_STOPPING = 5

class MetaDataset(Dataset):
    def __init__(self, finished_runs, start, end):
        size = len(finished_runs)
        self.data = finished_runs[int(start * size):int(end * size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        config_dict = self.data[idx]["config_dict"]
        x = [float(config_dict["seq_len"]),  # idk why, but needed
             config_dict["nb_channel"],
             config_dict["hidden_size"],
             int(config_dict["seq_stride_s"] * config_dict["fe"]),
             config_dict["nb_rnn_layers"],
             int(config_dict["window_size_s"] * config_dict["fe"]),
             config_dict["nb_conv_layers"],
             config_dict["stride_pool"],
             config_dict["stride_conv"],
             config_dict["kernel_conv"],
             config_dict["kernel_pool"],
             config_dict["dilation_conv"],
             config_dict["dilation_pool"]]
        x = torch.tensor(x)
        label = torch.tensor(self.data[idx]["cost_software"])
        return x, label


# run:

def run(config_dict):
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
    distribution_mode = config_dict["distribution_mode"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    net = PortiloopNetwork(config_dict).to(device=device_train)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr_adam, weight_decay=adam_w)

    net = net.train()
    # nb_weights = 0
    # for i in net.parameters():
    #     nb_weights += len(i)
    # has_envelope = 1
    # if config_dict["envelope_input"]:
    #     has_envelope = 2
    # config_dict["estimator_size_memory"] = nb_weights * window_size * seq_len * batch_size * has_envelope

    train_loader, validation_loader = generate_dataloader(window_size, fe, seq_len, seq_stride, distribution_mode, batch_size, nb_batch_per_epoch)

    best_model_loss_validation = 1
    best_model_epoch = 0
    early_stopping_cnt = 0

    h1_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    h2_zero = torch.zeros((nb_rnn_layers, batch_size, hidden_size), device=device_train)
    for epoch in range(nb_epoch_max):

        print(f"DEBUG: epoch: {epoch}")
        _t_start_train = time.time()

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
        _t_end_train = time.time()
        print(f"Training : {_t_end_train - _t_start_train} s")
        _t_start_validation = time.time()

        _, loss_validation, _, _, _ = get_accuracy_and_loss_pytorch(
            validation_loader, criterion, net, device_val, hidden_size, nb_rnn_layers)
        if loss_validation < best_model_loss_validation:
            best_model_loss_validation = loss_validation
            best_model_epoch = epoch
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
        _t_end_validation = time.time()
        print(f"Validation : {_t_end_validation - _t_start_validation} s")
        if early_stopping_cnt > NETWORK_EARLY_STOPPING:
            break
    return best_model_loss_validation, best_model_epoch


# hyperparameters

# batch_size_range_t = ["i", 256, 256]
# lr_adam_range_t = ["f", 0.0003, 0.0003]

seq_len_range_t = [10, 50, 5]  # min, max, step
kernel_conv_range_t = [3, 11, 2]  # min, max, step
kernel_pool_range_t = [3, 11, 2]  # min, max, step
stride_conv_range_t = [1, 5, 1]  # min, max, step
stride_pool_range_t = [1, 5, 1]  # min, max, step
dilation_conv_range_t = [1, 5, 1]  # min, max, step
dilation_pool_range_t = [1, 5, 1]  # min, max, step
nb_channel_range_t = [1, 70, 5]  # min, max, step
hidden_size_range_t = [2, 100, 5]  # min, max, step
window_size_s_range_t = [0.05, 1, 0.008]  # min, max, step
seq_stride_s_range_t = [0.05, 0.1, 0.004]  # min, max, step
nb_conv_layers_range_t = [1, 7, 1]  # min, max, step
nb_rnn_layers_range_t = [1, 5, 1]  # min, max, step


# dropout_range_t = ["f", 0.5, 0.5]
# first_layer_dropout_range_t = ["b", False, False]
# power_features_input_range_t = ["b", False, False]
# adam_w_range_t = ["f", 0.01, 0.01]


def clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


def sample_from_range(range_t, gaussian_mean=None, gaussian_std_factor=0.1):
    step = range_t[2]
    shift = range_t[0] % step
    min_t = round(range_t[0] / step)
    max_t = round(range_t[1] / step)
    diff_t = max_t - min_t
    gaussian_std = gaussian_std_factor * diff_t
    if gaussian_mean is None:
        res = random.uniform(min_t - 0.5, max_t + 0.5)  # otherwise extremum are less probable
    else:
        res = random.gauss(mu=gaussian_mean, sigma=gaussian_std)
        res = clip(res, min_t, max_t)
    res_unrounded = deepcopy(res) * step
    res = round(res)
    res *= step
    res += shift
    res = clip(res, range_t[0], range_t[1])
    res_unrounded = clip(res_unrounded, range_t[0], range_t[1])
    return res, res_unrounded


def same_config_dict(config1, config2):
    flag = 0
    if config1["seq_len"] != config2["seq_len"]:
        flag += 1
    if config1["nb_channel"] != config2["nb_channel"]:
        flag += 1
    if config1["hidden_size"] != config2["hidden_size"]:
        flag += 1
    if int(config1["seq_stride_s"] * config1["fe"]) != int(config2["seq_stride_s"] * config2["fe"]):
        flag += 1
    if config1["nb_rnn_layers"] != config2["nb_rnn_layers"]:
        flag += 1
    if int(config1["window_size_s"] * config1["fe"]) != int(config2["window_size_s"] * config2["fe"]):
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
    config_dict["distribution_mode"] = 1
    flag_in_exps = True
    while flag_in_exps:
        noise = random.choices(population=[True, False], weights=[EPSILON_NOISE, 1.0 - EPSILON_NOISE])[0]  # if we have already tried a config and lots of its neighbors, we will have a higher chance of getting a random config
        nb_out = 0
        std = 0.1
        if previous_exp == {} or noise:
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t)
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t)
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t)
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t)
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t)
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t)
        else:
            previous_unrounded = previous_exp["unrounded"]
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t, previous_unrounded["nb_conv_layers"])
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t, previous_unrounded["seq_len"])
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t, previous_unrounded["nb_channel"])
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t, previous_unrounded["hidden_size"])
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t, previous_unrounded["seq_stride_s"])
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t, previous_unrounded["nb_rnn_layers"])
        while nb_out < 1:

            if previous_exp == {} or noise:
                # sample completely randomly
                config_dict["window_size_s"], unrounded["window_size_s"] = sample_from_range(window_size_s_range_t, gaussian_std_factor=std)
                config_dict["stride_pool"], unrounded["stride_pool"] = sample_from_range(stride_pool_range_t, gaussian_std_factor=std)
                config_dict["stride_conv"], unrounded["stride_conv"] = sample_from_range(stride_conv_range_t, gaussian_std_factor=std)
                config_dict["kernel_conv"], unrounded["kernel_conv"] = sample_from_range(kernel_conv_range_t, gaussian_std_factor=std)
                config_dict["kernel_pool"], unrounded["kernel_pool"] = sample_from_range(kernel_pool_range_t, gaussian_std_factor=std)
                config_dict["dilation_conv"], unrounded["dilation_conv"] = sample_from_range(dilation_conv_range_t, gaussian_std_factor=std)
                config_dict["dilation_pool"], unrounded["dilation_pool"] = sample_from_range(dilation_pool_range_t, gaussian_std_factor=std)
            else:
                # sample gaussian from one of the previous experiments in the pareto front
                previous_unrounded = previous_exp["unrounded"]
                config_dict["window_size_s"], unrounded["window_size_s"] = sample_from_range(window_size_s_range_t, previous_unrounded["window_size_s"], gaussian_std_factor=std)
                config_dict["stride_pool"], unrounded["stride_pool"] = sample_from_range(stride_pool_range_t, previous_unrounded["stride_pool"], gaussian_std_factor=std)
                config_dict["stride_conv"], unrounded["stride_conv"] = sample_from_range(stride_conv_range_t, previous_unrounded["stride_conv"], gaussian_std_factor=std)
                config_dict["kernel_conv"], unrounded["kernel_conv"] = sample_from_range(kernel_conv_range_t, previous_unrounded["kernel_conv"], gaussian_std_factor=std)
                config_dict["kernel_pool"], unrounded["kernel_pool"] = sample_from_range(kernel_pool_range_t, previous_unrounded["kernel_pool"], gaussian_std_factor=std)
                config_dict["dilation_conv"], unrounded["dilation_conv"] = sample_from_range(dilation_conv_range_t, previous_unrounded["dilation_conv"], gaussian_std_factor=std)
                config_dict["dilation_pool"], unrounded["dilation_pool"] = sample_from_range(dilation_pool_range_t, previous_unrounded["dilation_pool"], gaussian_std_factor=std)
            std += 0.05
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
        for exp in all_exp:
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

        self.d1 = nn.Dropout(0)

        self.fc2 = nn.Linear(in_features=13*25,
                             out_features=13*25)

        self.d2 = nn.Dropout(0)

        self.fc3 = nn.Linear(in_features=13*25,
                             out_features=1)

    def to(self, device):
        super(SurrogateModel, self).to(device)
        self.device = device

    def forward(self, x):
        x_tensor = x.to(self.device)
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


def transform_config_dict_to_input(config_dict):
    x = [float(config_dict["seq_len"]),  # idk why, but needed
         config_dict["nb_channel"],
         config_dict["hidden_size"],
         int(config_dict["seq_stride_s"] * config_dict["fe"]),
         config_dict["nb_rnn_layers"],
         int(config_dict["window_size_s"] * config_dict["fe"]),
         config_dict["nb_conv_layers"],
         config_dict["stride_pool"],
         config_dict["stride_conv"],
         config_dict["kernel_conv"],
         config_dict["kernel_pool"],
         config_dict["dilation_conv"],
         config_dict["dilation_pool"]]
    x = torch.tensor(x)
    return x


def train_surrogate(net, all_experiments):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0, dampening=0, weight_decay=0.01, nesterov=False)
    criterion = nn.MSELoss()
    best_val_loss = np.inf
    best_model = None
    early_stopping_counter = 0
    random.shuffle(all_experiments)

    for epoch in range(MAX_META_EPOCHS):
        if len(all_experiments) > START_META_TRAIN_VAL_AFTER:
            train_dataset = MetaDataset(all_experiments, start=0, end=META_TRAIN_VAL_RATIO)
            validation_dataset = MetaDataset(all_experiments, start=META_TRAIN_VAL_RATIO, end=1)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        else:
            train_dataset = MetaDataset(all_experiments, start=0, end=1)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        losses = []

        net.train()
        for batch_data in train_loader:
            batch_samples, batch_labels = batch_data
            batch_samples = batch_samples.to(device=META_MODEL_DEVICE).float()
            batch_labels = batch_labels.to(device=META_MODEL_DEVICE).float()

            optimizer.zero_grad()
            output = net(batch_samples)
            output = output.view(-1)
            loss = criterion(output, batch_labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()

        mean_loss = np.mean(losses)
        print(f"DEBUG: epoch {epoch} mean_loss_training = {mean_loss}")

        if len(all_experiments) > START_META_TRAIN_VAL_AFTER:
            net.eval()
            losses = []
            with torch.no_grad():
                for batch_data in validation_loader:
                    batch_samples, batch_labels = batch_data
                    batch_samples = batch_samples.to(device=META_MODEL_DEVICE).float()
                    batch_labels = batch_labels.to(device=META_MODEL_DEVICE).float()

                    output = net(batch_samples)
                    output = output.view(-1)
                    loss = criterion(output, batch_labels)
                    losses.append(loss.item())

                mean_loss_validation = np.mean(losses)
                print(f"DEBUG: mean_loss_validation = {mean_loss_validation}")
                if mean_loss_validation < best_val_loss:
                    best_val_loss = mean_loss_validation
                    early_stopping_counter = 0
                    best_model = deepcopy(net)
                else:
                    early_stopping_counter += 1
                # early stopping:
                if early_stopping_counter >= META_EARLY_STOPPING:
                    net = best_model
                    mean_loss = best_val_loss
                    print(f"DEBUG: meta training converged at epoch:{epoch} (-{META_EARLY_STOPPING})")
                    break
                elif epoch == MAX_META_EPOCHS - 1:
                    print(f"DEBUG: meta training did not converge after epoch:{epoch}")
                    break
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
    best_cost_software = 1
    for exp in all_experiments:
        if exp["cost_software"] < experiment["cost_software"] and exp["cost_hardware"] < experiment["cost_hardware"]:
            nb_dominating += 1
        if exp["cost_software"] > experiment["cost_software"] and exp["cost_hardware"] > experiment["cost_hardware"]:
            nb_dominated += 1
        if exp["cost_software"] < best_cost_software:
            best_cost_software = exp["cost_software"]

    score_not_dominated = 1.0 - float(nb_dominating) / len(all_experiments)
    score_dominating = nb_dominated / len(all_experiments)
    score_distance_from_best_loss = best_cost_software / experiment[
        "cost_software"]  # The lower is the predicted experiment loss, the better. This score is close to 1 when you reach a loss as good as the lowest one of all exp. If yours is better, then the score will be above 1. Otherwise the farest you are, the lower is your score
    return score_dominating + score_not_dominated + 2*score_distance_from_best_loss

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
            nerf = 0
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


def dump_network_files(finished_experiments, pareto_front):
    """
    exports pickled files to path_pareto
    """
    path_current_finished = path_pareto / (RUN_NAME + "_finished.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    #   path_current_launched = path_pareto / (RUN_NAME + "_launched.pkl")
    with open(path_current_finished, "wb") as f:
        pkl.dump(finished_experiments, f)
    #  with open(path_current_launched, "wb") as f:
    #     pkl.dump(launched_experiments, f)
    with open(path_current_pareto, "wb") as f:
        pkl.dump(pareto_front, f)


def load_network_files():
    """
    loads pickled files from path_pareto
    returns None, None if not found
    else returns all_experiments, pareto_front
    """
    path_current_finished = path_pareto / (RUN_NAME + "_finished.pkl")
    path_current_pareto = path_pareto / (RUN_NAME + "_pareto.pkl")
    #  path_current_launched = path_pareto / (RUN_NAME + "_launched.pkl")
    if not path_current_finished.exists() or not path_current_pareto.exists():
        return None, None
    with open(path_current_finished, "rb") as f:
        finished_experiments = pkl.load(f)
    with open(path_current_pareto, "rb") as f:
        pareto_front = pkl.load(f)
    #  with open(path_current_launched, "rb") as f:
    #     launched_experiments = pkl.load(f)
    return finished_experiments, pareto_front


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


def iterative_training_local():
    logger = LoggerWandbPareto(RUN_NAME)

    all_experiments, pareto_front = load_network_files()

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
            if nb_params > MAX_NB_PARAMETERS or nb_params < MIN_NB_PARAMETERS:
                continue
            if nb_params < MIN_NB_PARAMETERS:
                print("ERROR")
            with torch.no_grad():
                input = transform_config_dict_to_input(config_dict)
                predicted_loss = meta_model(input).item()

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

        dump_network_files(all_experiments, pareto_front)
        logger.log(surrogate_loss=meta_loss, surprise=surprise, all_experiments=all_experiments, pareto_front=pareto_front)

    print(f"End of meta-training.")


# Main:

if __name__ == "__main__":
    iterative_training_local()
