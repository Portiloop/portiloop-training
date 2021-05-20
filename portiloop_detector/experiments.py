import logging

import torch
from torch import nn

from portiloop_detector_training import PortiloopNetwork, generate_dataloader, path_dataset, run_inference, get_metrics



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

    output_test, labels_test, loss_test, accuracy_test, tp, tn, fp, fn = run_inference(test_loader, criterion, net, device_val, hidden_size, nb_rnn_layers, classification, batch_size_test)
    f1_test, precision_test, recall_test = get_metrics(tp, fp, fn)


np.savetxt(portiloop.path_dataset / "labels_portiloop.txt", state)
