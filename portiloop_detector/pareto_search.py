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

EPSILON_NOISE = 0.1  # a completely random model will be selected this portion of the time, otherwise, it is sampled as a gaussian from the pareto front
EPSILON_EXP_NOISE = 0.1

MAX_NB_PARAMETERS = 100000  # everything over this number of parameters will be discarded
MAX_LOSS = 0.1  # to normalize distances

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

NB_BATCH_PER_EPOCH = 10000

RUN_NAME = "pareto_search_5"

NB_SAMPLED_MODELS_PER_ITERATION = 200  # number of models sampled per iteration, only the best predicted one is selected


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
lr_adam_range_t = ["f", 0.0003, 0.0003]

seq_len_range_t = ["i", 10, 50]

kernel_conv_range_t = ["i", 3, 9]
kernel_pool_range_t = ["i", 3, 5]
stride_conv_range_t = ["i", 1, 3]
stride_pool_range_t = ["i", 1, 3]
dilation_conv_range_t = ["i", 1, 5]
dilation_pool_range_t = ["i", 1, 5]
nb_channel_range_t = ["i", 10, 50]
hidden_size_range_t = ["i", 2, 100]
dropout_range_t = ["f", 0.5, 0.5]
window_size_s_range_t = ["f", 0.05, 0.1]
seq_stride_s_range_t = ["f", 0.05, 0.1]
nb_conv_layers_range_t = ["i", 1, 7]
nb_rnn_layers_range_t = ["i", 1, 5]
# first_layer_dropout_range_t = ["b", False, False]
# power_features_input_range_t = ["b", False, False]
adam_w_range_t = ["f", 0.01, 0.01]


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


def sample_config_dict(name, previous_exp):
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

    nb_out = 0
    while nb_out < 1:

        if previous_exp == {} or noise:
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
            # previous_experiment = random.choice(pareto_front)
            previous_unrounded = previous_exp["unrounded"]
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

        self.d1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=200,
                             out_features=200)

        self.d2 = nn.Dropout(0.5)

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

        x_tensor = F.relu(self.d1(self.fc1(x_tensor)))
        x_tensor = F.relu(self.d2(self.fc2(x_tensor)))
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
            targ = torch.tensor([labels[i], ])
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
    plt.clf()
    # all experiments:
    x_axis = [exp["cost_hardware"] * MAX_NB_PARAMETERS for exp in all_experiments]
    y_axis = [exp["cost_software"] * MAX_LOSS for exp in all_experiments]
    plt.plot(x_axis, y_axis, 'bo')
    # pareto:
    x_axis = [exp["cost_hardware"] * MAX_NB_PARAMETERS for exp in ordered_pareto_front]
    y_axis = [exp["cost_software"] * MAX_LOSS for exp in ordered_pareto_front]
    plt.plot(x_axis, y_axis, 'ro-')
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
    return np.array([experiment["cost_software"], experiment["cost_hardware"]])


def pareto_efficiency(experiment, pareto_front, histo):
    if len(pareto_front) < 2:
        return 0.0
    v_p = vector_exp(experiment)
    dominates = True
    all_dists = []
    for i in range(len(pareto_front)):
        exp = pareto_front[i]
        if exp["cost_software"] <= experiment["cost_software"] and exp["cost_hardware"] <= experiment["cost_hardware"]:
            dominates = False
        if i < len(pareto_front) - 1:
            next = pareto_front[i + 1]
            v_a = vector_exp(exp)
            v_b = vector_exp(next)
            dist = dist_p_to_ab(v_a, v_b, v_p)
            all_dists.append(dist)
    assert len(all_dists) >= 1
    res = min(all_dists)  # distance to pareto
    if not dominates:
        res *= -1.0
    # subtract density around number of parameters
    if not (experiment["cost_hardware"] > histo[1][-1] or experiment["cost_hardware"] < histo[1][0]):
        idx = np.where(histo[1] > experiment["cost_hardware"])[0][0] - 1
        res -= histo[0][idx]
    return res


def exp_max_pareto_efficiency(experiments, pareto_front, all_experiments):
    assert len(experiments) >= 1
    noise = random.choices(population=[True, False], weights=[EPSILON_EXP_NOISE, 1.0 - EPSILON_EXP_NOISE])[0]
    if noise or len(pareto_front) == 0:
        return random.choice(experiments)
    else:
        assert len(all_experiments) != 0
        histo = np.histogram([exp["cost_hardware"] for exp in all_experiments], bins=10)
        max_efficiency = -np.inf
        best_exp = None
        for exp in experiments:
            efficiency = pareto_efficiency(exp, pareto_front, histo)
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
        print(f"DEBUG: no meta dataset found, starting new run")
        all_experiments = []  # list of dictionaries
        pareto_front = []  # list of dictionaries, subset of all_experiments
    else:
        print(f"DEBUG: existing meta dataset loaded")

    meta_model = SurrogateModel()
    meta_model.to(META_MODEL_DEVICE)

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
            config_dict, unrounded = sample_config_dict(name=RUN_NAME + "_" + str(num_experiment), previous_exp=prev_exp)

            nb_params = nb_parameters(config_dict)
            if nb_params > MAX_NB_PARAMETERS:
                continue

            with torch.no_grad():
                predicted_loss = meta_model(config_dict).item()

            exp["cost_hardware"] = nb_params / MAX_NB_PARAMETERS
            exp["cost_software"] = predicted_loss / MAX_LOSS
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

        print(f"nb parameters: {nb_params * MAX_NB_PARAMETERS}")
        print(f"predicted loss: {predicted_loss * MAX_LOSS}")
        print("training...")

        exp["cost_software"] = run(config_dict) / MAX_LOSS

        pareto_front = update_pareto(exp, pareto_front)
        all_experiments.append(exp)

        prev_exp = exp

        print(f"actual loss: {exp['cost_software'] * MAX_LOSS}")
        surprise = exp['cost_software'] - predicted_loss
        print(f"surprise: {surprise * MAX_LOSS}")
        print("training surrogate model...")

        meta_model.train()
        meta_model, mean_loss = train_surrogate(meta_model, all_experiments)

        print(f"surrogate model loss: {mean_loss}")

        dump_files(all_experiments, pareto_front)
        logger.log(surrogate_loss=mean_loss, surprise=surprise, all_experiments=all_experiments, pareto_front=pareto_front)

    print(f"End of meta-training.")
