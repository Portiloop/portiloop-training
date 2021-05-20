import logging

import torch
from torch import nn

from portiloop_detector_training import PortiloopNetwork, generate_dataloader, path_dataset


def run_test(config_dict):
    logging.debug(f"config_dict: {config_dict}")
    experiment_name = config_dict['experiment_name']
    window_size_s = config_dict["window_size_s"]
    fe = config_dict["fe"]
    seq_stride_s = config_dict["seq_stride_s"]
    hidden_size = config_dict["hidden_size"]
    device_val = config_dict["device_val"]
    device_train = config_dict["device_train"]
    nb_rnn_layers = config_dict["nb_rnn_layers"]
    classification = config_dict["classification"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    torch.seed()
    net = PortiloopNetwork(config_dict).to(device=device_val)
    criterion = nn.MSELoss() if not classification else nn.BCELoss()

    _, _, _, _, test_loader, batch_size_test = generate_dataloader(window_size, fe, None, seq_stride, None, None, None)

    checkpoint = torch.load(path_dataset / experiment_name)
    logging.debug("Use checkpoint model")
    net.load_state_dict(checkpoint['model_state_dict'])


with torch.no_grad():
    for idx in range(0, 250 * 60*45, seq_stride):
        batch_samples_input1, batch_samples_input2, batch_samples_input3, batch_labels = ds_test[idx]
        batch_samples_input1 = batch_samples_input1.to(device=device_val).float().view(1, 1, -1)
        batch_samples_input2 = batch_samples_input2.to(device=device_val).float().view(1, 1, -1)
        batch_samples_input3 = batch_samples_input3.to(device=device_val).float().view(1, 1, -1)
        batch_labels = batch_labels.to(device=device_val).float()
        output, h1, h2 = net(batch_samples_input1, batch_samples_input2, batch_samples_input3, h1, h2)
        output = output.view(-1)

        if output > THRESHOLD:
            if batch_labels > THRESHOLD:
                state.append(1)
            else:
                state.append(2)
        else:
            if batch_labels > THRESHOLD:
                state.append(3)
            else:
                state.append(4)

np.savetxt(portiloop.path_dataset / "labels_portiloop.txt", state)
