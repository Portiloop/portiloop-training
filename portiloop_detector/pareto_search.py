"""
Pareto-optimal hyperparameter search (meta-learning)
"""
import os
import pickle as pkl
# all imports
import random
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import wandb
from portiloop_detector_training import PortiloopNetwork, run
from utils import MAX_NB_PARAMETERS, EPSILON_EXP_NOISE, sample_config_dict, MIN_NB_PARAMETERS, MAXIMIZE_F1_SCORE

# all constants (no hyperparameters here!)

THRESHOLD = 0.2
WANDB_PROJECT_PARETO = "pareto"

path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
path_pareto = Path(__file__).absolute().parent.parent / 'pareto'

# path = "/content/drive/MyDrive/Data/MASS/"
# path_dataset = Path(path)
# path_pareto = Path("/content/drive/MyDrive/Data/pareto_results/")

MAX_META_ITERATIONS = 1000  # maximum number of experiments

# MAX_LOSS = 0.1  # to normalize distances

META_MODEL_DEVICE = "cpu"  # the surrogate model will be trained on this device

RUN_NAME = "pareto_search_11"

NB_SAMPLED_MODELS_PER_ITERATION = 200  # number of models sampled per iteration, only the best predicted one is selected

DEFAULT_META_EPOCHS = 100  # default number of meta-epochs before entering meta train/val training regime
START_META_TRAIN_VAL_AFTER = 100  # minimum number of experiments in the dataset before using a validation set
META_TRAIN_VAL_RATIO = 0.8  # portion of experiments in meta training sets
MAX_META_EPOCHS = 500  # surrogate training will stop after this number of meta-training epochs if the model doesn't converge
META_EARLY_STOPPING = 30  # meta early stopping after this number of unsuccessful meta epochs


class MetaDataset(Dataset):
    def __init__(self, finished_runs, start, end):
        size = len(finished_runs)
        self.data = finished_runs[int(start * size):int(end * size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        config_dict = self.data[idx]["config_dict"]
        x = transform_config_dict_to_input(config_dict)
        label = torch.tensor(self.data[idx]["cost_software"])
        return x, label


def nb_parameters(config_dict):
    net = PortiloopNetwork(config_dict)
    res = sum(p.numel() for p in net.parameters())
    del net
    return res


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        nb_features = 17
        self.fc1 = nn.Linear(in_features=nb_features,  # nb hyperparameters
                             out_features=nb_features * 25)  # in SMBO paper : 25 * hyperparameters... Seems huge

        self.d1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=nb_features * 25,
                             out_features=nb_features * 25)

        self.d2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=nb_features * 25,
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
         config_dict["dilation_pool"],
         int(config_dict['RNN']),
         int(config_dict['envelope_input']),
         config_dict["lr_adam"],
         config_dict["batch_size"]]
    x = torch.tensor(x)
    return x


def train_surrogate(net, all_experiments):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0.01, nesterov=False)
    criterion = nn.MSELoss()
    best_val_loss = np.inf
    best_model = None
    early_stopping_counter = 0
    random.shuffle(all_experiments)
    max_epoch = MAX_META_EPOCHS if len(all_experiments) > START_META_TRAIN_VAL_AFTER else len(all_experiments)

    for epoch in range(max_epoch):
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
        # print(f"DEBUG: epoch {epoch} mean_loss_training = {mean_loss}")

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
                # print(f"DEBUG: mean_loss_validation = {mean_loss_validation}")
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
    plt.ylabel(f"validation cost")
    # plt.ylim(top=0.1)
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


# def vector_exp(experiment):
#     return np.array([experiment["cost_software"] / MAX_LOSS, experiment["cost_hardware"] / MAX_NB_PARAMETERS])
#

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
    score_distance_from_best_loss = abs(best_cost_software / experiment[
        "cost_software"])  # The lower is the predicted experiment loss, the better. This score is close to 1 when you reach a loss as good as the lowest one of all exp. If yours is better, then the score will be above 1. Otherwise the farest you are, the lower is your score
    return score_dominating + score_not_dominated + score_distance_from_best_loss

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
            nerf = histo[0][min(idx, len(histo[0]) - 1)] * MAX_NB_PARAMETERS
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
        self.wandb_run = wandb.init(project=WANDB_PROJECT_PARETO, entity="portiloop", id=run_name, resume="allow", reinit=True)

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
                predicted_cost = meta_model(input).item()

            exp["cost_hardware"] = nb_params
            exp["cost_software"] = predicted_cost
            exp["config_dict"] = config_dict
            exp["unrounded"] = unrounded

            exps.append(exp)

            if len(exps) >= NB_SAMPLED_MODELS_PER_ITERATION:
                # select model
                model_selected = True
                exp = exp_max_pareto_efficiency(exps, pareto_front, all_experiments)

        config_dict = exp["config_dict"]
        predicted_cost = exp["cost_software"]
        nb_params = exp["cost_hardware"]

        print(f"config: {config_dict}")

        print(f"nb parameters: {nb_params}")
        print(f"predicted cost: {predicted_cost}")
        print("training...")
        best_loss, best_f1_score, exp["best_epoch"] = run(exp["config_dict"], WANDB_PROJECT_PARETO + "_runs_11", save_model=False, unique_name=True)
        exp["cost_software"] = 1 - best_f1_score if MAXIMIZE_F1_SCORE else best_loss

        pareto_front = update_pareto(exp, pareto_front)
        all_experiments.append(exp)

        prev_exp = exp

        print(f"actual cost: {exp['cost_software']}")
        surprise = exp['cost_software'] - predicted_cost
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
