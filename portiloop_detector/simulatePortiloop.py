# script that will test the network like if it was on the device
import logging

import torch
from torch import nn

from portiloop_detector.experiments import path_experiment
from portiloop_detector.portiloop_detector_training import PortiloopNetwork, generate_dataloader, run_inference

FPGA_NN_EXEC_TIME = 10  # equivalent to 40 ms
ERROR_FPGA_EXEC_TIME = 3  # to be sure there is no overlap


def simulate(c_dict):
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
    split_idx = c_dict["split_idx"]
    window_size = int(window_size_s * fe)
    seq_stride = int(seq_stride_s * fe)

    nb_parallel_runs = seq_stride // (FPGA_NN_EXEC_TIME + ERROR_FPGA_EXEC_TIME)
    logging.debug(f"nb_parallel_runs: {nb_parallel_runs}")
    stride_between_runs = seq_stride // nb_parallel_runs
    logging.debug(f"stride_between_runs: {stride_between_runs}")

    if device_val.startswith("cuda") or device_train.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"

    torch.seed()
    net = PortiloopNetwork(c_dict).to(device=device_val)
    criterion = nn.MSELoss() if not classification else nn.BCELoss()

    _, _, _, test_loader, batch_size_test, test_subject = generate_dataloader(window_size=window_size, fe=fe, seq_len=None, seq_stride=seq_stride,
                                                                              distribution_mode=None, batch_size=None, nb_batch_per_epoch=None,
                                                                              classification=classification, split_i=split_idx,
                                                                              divider=nb_parallel_runs)
    # with open(path_experiment / "testloader.pkl", 'wb') as file:
    #     pickle.dump(test_loader, file)
    checkpoint = torch.load(path_experiment / experiment_name)
    logging.debug("Use trained model")
    net.load_state_dict(checkpoint['model_state_dict'])

    output_test, labels_test, loss_test, accuracy_test, tp, tn, fp, fn = run_inference(test_loader, criterion, net, device_val, hidden_size,
                                                                                       nb_rnn_layers, classification, batch_size_test, max_value=0)


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    simulate()
