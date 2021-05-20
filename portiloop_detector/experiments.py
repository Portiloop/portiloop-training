import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import nn

from portiloop_detector_training import PortiloopNetwork, generate_dataloader, path_dataset, run_inference, get_metrics, get_config_dict

path_experiment = Path(__file__).absolute().parent.parent / 'experiments'

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

    _, _, _, test_loader, batch_size_test = generate_dataloader(window_size=window_size, fe=fe, seq_len=None, seq_stride=seq_stride,
                                                                distribution_mode=None, batch_size=None, nb_batch_per_epoch=None)

    checkpoint = torch.load(path_experiment / experiment_name)
    logging.debug("Use trained model")
    net.load_state_dict(checkpoint['model_state_dict'])

    output_test, labels_test, loss_test, accuracy_test, tp, tn, fp, fn = run_inference(test_loader, criterion, net, device_val, hidden_size,
                                                                                       nb_rnn_layers, classification, batch_size_test)
    f1_test, precision_test, recall_test = get_metrics(tp, fp, fn)
    logging.debug(f"accuracy_test = {accuracy_test}")
    logging.debug(f"loss_test = {loss_test}")
    logging.debug(f"f1_test = {f1_test}")
    logging.debug(f"precision_test = {precision_test}")
    logging.debug(f"recall_test = {recall_test}")

    state = tp + fp * 2 + tn * 3 + fn * 4
    state = np.hstack(np.transpose(np.split(state.cpu().detach().numpy(), batch_size_test)))

    np.savetxt(path_experiment / f"labels_{experiment_name}.txt", state)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--experiment_index', type=str, default=0)
    args = parser.parse_args()

    exp_index = args.experiment_index

    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s', filename=args.output_file, level=logging.DEBUG)
        logging.debug('This message should go to the log file')
        logging.info('So should this')
        logging.warning('And this, too')
        logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    config_dict = get_config_dict(exp_index)
    config_dict["experiment_name"] = "test_v3_1621522310727862534"

    run_test(config_dict)
