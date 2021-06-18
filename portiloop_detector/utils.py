import logging
from copy import deepcopy
from math import floor
from random import choices, uniform, gauss

MIN_NB_PARAMETERS = 1000  # everything below this number of parameters will be discarded
MAX_NB_PARAMETERS = 80000  # everything over this number of parameters will be discarded

NB_BATCH_PER_EPOCH = 1000
MAX_EPOCHS_PER_EXPERIMENT = 150  # experiments are evaluated after this number of epoch by the meta learner

EPSILON_NOISE = 0.25  # a completely random model will be selected this portion of the time, otherwise, it is sampled from a gaussian
EPSILON_EXP_NOISE = 0.1  # a random experiment is selected within all sampled experiments this portion of the time
NETWORK_EARLY_STOPPING = 20

seq_len_range_t = [50, 50, 1]  # min, max, step
kernel_conv_range_t = [3, 11, 2]  # min, max, step
kernel_pool_range_t = [3, 11, 2]  # min, max, step
stride_conv_range_t = [1, 1, 1]  # min, max, step
stride_pool_range_t = [1, 1, 1]  # min, max, step
dilation_conv_range_t = [1, 1, 1]  # min, max, step
dilation_pool_range_t = [1, 1, 1]  # min, max, step
nb_channel_range_t = [1, 100, 5]  # min, max, step
hidden_size_range_t = [2, 100, 5]  # min, max, step
window_size_s_range_t = [0.05, 0.5, 0.008]  # min, max, step
seq_stride_s_range_t = [0.05, 0.2, 0.008]  # min, max, step
nb_conv_layers_range_t = [1, 5, 1]  # min, max, step
nb_rnn_layers_range_t = [1, 2, 1]  # min, max, step
rnn_range_t = [1, 1, 1]
envelope_input_range_t = [0, 0, 1]
lr_adam_range_t = [0.0003, 0.0009, 0.0002]
batch_size_range_t = [64, 256, 64]

PROFILE_META = False


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
        res = uniform(min_t - 0.5, max_t + 0.5)  # otherwise extremum are less probable
    else:
        res = gauss(mu=gaussian_mean, sigma=gaussian_std)
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
    if config1["RNN"] != config2["RNN"]:
        flag += 1
    if config1["envelope_input"] != config2["envelope_input"]:
        flag += 1
    if config1["lr_adam"] != config2["lr_adam"]:
        flag += 1
    if config1["batch_size"] != config2["batch_size"]:
        flag += 1
    return flag == 0


def sample_config_dict(name, previous_exp, all_exp):
    config_dict = dict(experiment_name=name,
                       device_train="cuda:0",
                       device_val="cuda:0",
                       nb_epoch_max=MAX_EPOCHS_PER_EXPERIMENT,
                       max_duration=int(71.5 * 3600),
                       nb_epoch_early_stopping_stop=NETWORK_EARLY_STOPPING,
                       early_stopping_smoothing_factor=0.1,
                       fe=250,
                       nb_batch_per_epoch=NB_BATCH_PER_EPOCH)

    unrounded = {}

    # constant things:

    # config_dict["RNN"] = True
    # config_dict["envelope_input"] = True
    # config_dict["batch_size"] = 256
    config_dict["first_layer_dropout"] = False
    config_dict["power_features_input"] = False
    config_dict["dropout"] = 0.5
    # config_dict["lr_adam"] = 0.0003
    config_dict["adam_w"] = 0.01
    config_dict["distribution_mode"] = 0
    config_dict["classification"] = True
    config_dict["reg_balancing"] = 'none'
    config_dict["split_idx"] = 0
    config_dict["validation_divider"] = 10

    flag_in_exps = True
    while flag_in_exps:
        noise = choices(population=[True, False], weights=[EPSILON_NOISE, 1.0 - EPSILON_NOISE])[0]  # if we have already tried a config and lots of its neighbors, we will have a higher chance of getting a random config
        nb_out = 0
        std = 0.1
        if previous_exp == {} or noise:
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t)
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t)
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t)
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t)
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t)
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t)
            config_dict["RNN"], unrounded["RNN"] = sample_from_range(rnn_range_t)
            config_dict["RNN"] = config_dict["RNN"] == 1
            config_dict["envelope_input"], unrounded["envelope_input"] = sample_from_range(envelope_input_range_t)
            config_dict["envelope_input"] = config_dict["envelope_input"] == 1
            config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t)
            config_dict["batch_size"], unrounded["batch_size"] = sample_from_range(batch_size_range_t)
        else:
            previous_unrounded = previous_exp["unrounded"]
            if 'RNN' not in previous_unrounded.keys():
                previous_unrounded['RNN'] = 0.5
            if 'envelope_input' not in previous_unrounded.keys():
                previous_unrounded['envelope_input'] = 0.5
            config_dict["nb_conv_layers"], unrounded["nb_conv_layers"] = sample_from_range(nb_conv_layers_range_t, previous_unrounded["nb_conv_layers"])
            config_dict["seq_len"], unrounded["seq_len"] = sample_from_range(seq_len_range_t, previous_unrounded["seq_len"])
            config_dict["nb_channel"], unrounded["nb_channel"] = sample_from_range(nb_channel_range_t, previous_unrounded["nb_channel"])
            config_dict["hidden_size"], unrounded["hidden_size"] = sample_from_range(hidden_size_range_t, previous_unrounded["hidden_size"])
            config_dict["seq_stride_s"], unrounded["seq_stride_s"] = sample_from_range(seq_stride_s_range_t, previous_unrounded["seq_stride_s"])
            config_dict["nb_rnn_layers"], unrounded["nb_rnn_layers"] = sample_from_range(nb_rnn_layers_range_t, previous_unrounded["nb_rnn_layers"])
            config_dict["RNN"], unrounded["RNN"] = sample_from_range(rnn_range_t, previous_unrounded['RNN'])
            config_dict["RNN"] = config_dict["RNN"] == 1
            config_dict["envelope_input"], unrounded["envelope_input"] = sample_from_range(envelope_input_range_t, previous_unrounded['envelope_input'])
            config_dict["envelope_input"] = config_dict["envelope_input"] == 1
            config_dict["lr_adam"], unrounded["lr_adam"] = sample_from_range(lr_adam_range_t, previous_unrounded['lr_adam'])
            config_dict["batch_size"], unrounded["batch_size"] = sample_from_range(batch_size_range_t, previous_unrounded['batch_size'])
        config_dict["seq_len"] = 1 if not config_dict["RNN"] else config_dict["seq_len"]
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
                logging.debug(f"DEBUG : config already tried = {config_dict}")
                break

    config_dict["nb_out"] = nb_out
    config_dict["time_in_past"] = config_dict["seq_len"] * config_dict["seq_stride_s"]

    return config_dict, unrounded


def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
