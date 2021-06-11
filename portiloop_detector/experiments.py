import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import nn

from portiloop_detector_training import PortiloopNetwork, generate_dataloader, run_inference, get_metrics, get_config_dict, PHASE

path_experiment = Path(__file__).absolute().parent.parent / 'experiments'


def run_test(c_dict):
    logging.debug(f"config_dict: {c_dict}")
    experiment_name = c_dict['experiment_name']
    window_size_s = c_dict["window_size_s"]
    fe = c_dict["fe"]
    seq_stride_s = c_dict["seq_stride_s"]
    hidden_size = c_dict["hidden_size"]
    device_val = c_dict["device_val"]
    device_train = c_dict["device_train"]
    nb_rnn_layers = c_dict["nb_rnn_layers"]
    classification = c_dict["classification"]

    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    torch.seed()
    net = PortiloopNetwork(c_dict).to(device=device_val)
    criterion = nn.MSELoss() if not classification else nn.BCELoss()

    _, _, _, test_loader, batch_size_test, test_subject = generate_dataloader(window_size=window_size, fe=fe, seq_len=None, seq_stride=seq_stride,
                                                                              distribution_mode=None, batch_size=None, nb_batch_per_epoch=None,
                                                                              classification=classification)
    # with open(path_experiment / "testloader.pkl", 'wb') as file:
    #     pickle.dump(test_loader, file)
    checkpoint = torch.load(path_experiment / experiment_name)
    logging.debug("Use trained model")
    net.load_state_dict(checkpoint['model_state_dict'])

    output_test, labels_test, loss_test, accuracy_test, tp, tn, fp, fn = run_inference(test_loader, criterion, net, device_val, hidden_size,
                                                                                       nb_rnn_layers, classification, batch_size_test, max_value=0)
    f1_test, precision_test, recall_test = get_metrics(tp, fp, fn)
    logging.debug(f"accuracy_test = {accuracy_test}")
    logging.debug(f"loss_test = {loss_test}")
    logging.debug(f"f1_test = {f1_test}")
    logging.debug(f"precision_test = {precision_test}")
    logging.debug(f"recall_test = {recall_test}")

    state = tp + fp * 2 + tn * 3 + fn * 4
    state = np.transpose(np.split(state.cpu().detach().numpy(), len(state) / batch_size_test))
    labels_test = np.transpose(np.split(labels_test.cpu().detach().numpy(), len(labels_test) / batch_size_test))
    output_test = np.transpose(np.split(output_test.cpu().detach().numpy(), len(output_test) / batch_size_test))
    return f1_test, precision_test, recall_test
    # np.savetxt(path_experiment / f"labels_{experiment_name}_{PHASE}.txt", state)
    # np.savetxt(path_experiment / f"subject_{experiment_name}_{PHASE}.txt", test_subject, format="%s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--experiment_index', type=int, default=0)
    parser.add_argument('--max_split', type=int, default=10)
    args = parser.parse_args()

    max_split = args.max_split
    exp_index = args.experiment_index

    if args.output_file is not None:
        logging.basicConfig(format='%(levelname)s: %(message)s', filename=args.output_file, level=logging.DEBUG)
        logging.debug('This message should go to the log file')
        logging.info('So should this')
        logging.warning('And this, too')
        logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    res = []
    config_dict = dict()
    for split_idx in range(max_split):
        config_dict = get_config_dict(exp_index, split_idx)
        config_dict["experiment_name"] = ""
        res.append(run_test(config_dict))
    res = np.array(res)
    std_f1_test, std_precision_test, std_recall_test = np.std(res, axis=0)
    mean_f1_test, mean_precision_test, mean_recall_test = np.mean(res, axis=0)
    print(config_dict["experiment_name"])
    print(f"Recall: {mean_recall_test} + {std_recall_test}")
    print(f"Precision: {mean_precision_test} + {std_precision_test}")
    print(f"f1: {mean_f1_test} + {std_f1_test}")
