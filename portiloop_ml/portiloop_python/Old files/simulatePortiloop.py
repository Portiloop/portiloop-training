# script that will test the network like if it was on the device
import logging

import numpy as np
import torch
from torch import nn

from portiloop_detector.experiments import path_experiment
from portiloop_detector.portiloop_detector_training import PortiloopNetwork, generate_dataloader, run_inference, get_metrics, get_config_dict

FPGA_NN_EXEC_TIME = 5  # equivalent to 40 ms
ERROR_FPGA_EXEC_TIME = 0  # to be sure there is no overlap


def simulate(c_dict, w, split_idx):
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

    nb_parallel_runs = seq_stride // (FPGA_NN_EXEC_TIME + ERROR_FPGA_EXEC_TIME)
    print(f"seq_stride: {seq_stride}")
    print(f"FPGA_NN_EXEC_TIME + ERROR_FPGA_EXEC_TIME: {FPGA_NN_EXEC_TIME + ERROR_FPGA_EXEC_TIME}")
    print(f"nb_parallel_runs: {nb_parallel_runs}")
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
                                                                              network_stride=stride_between_runs)

    checkpoint = torch.load(path_experiment / experiment_name, map_location=torch.device(device_val))
    logging.debug("Use trained model")
    net.load_state_dict(checkpoint['model_state_dict'])

    output_test, labels_test, loss_test, accuracy_test, tp, tn, fp, fn = run_inference(test_loader, criterion, net, device_val, hidden_size,
                                                                                       nb_rnn_layers, classification, batch_size_test, max_value=0)

    nb_segment_test = len(np.hstack([range(int(s[1]), int(s[2])) for s in test_subject]))
    labels_test = np.transpose(np.split(labels_test.cpu().detach().numpy(), len(labels_test) / batch_size_test))
    output_test = np.transpose(np.split(output_test.cpu().detach().numpy(), len(output_test) / batch_size_test))
    logging.debug(f"shape output test: {output_test.shape}")
    logging.debug(f"nb_segment_test: {nb_segment_test}")
    output_segments = []
    for s in range(nb_segment_test):
        output_segments.append(zip(*(output_test[s * nb_parallel_runs + i] for i in range(nb_parallel_runs))))
        output_segments[-1] = np.hstack(np.array([list(a) for a in output_segments[-1]]))
    output_portiloop = np.hstack(np.array(output_segments))
    labels_segments = []
    for s in range(nb_segment_test):
        labels_segments.append(zip(*(labels_test[s * nb_parallel_runs + i] for i in range(nb_parallel_runs))))
        labels_segments[-1] = np.hstack(np.array([list(a) for a in labels_segments[-1]]))
    labels_portiloop = np.hstack(np.array(labels_segments))

    # new_output = [0]
    # for i in range(1,len(output_portiloop)):
    #     val = 0
    #     if output_portiloop[i] == 1 and output_portiloop[i-1] == 1:
    #         val = 1
    #     new_output.append(val)
    # output_portiloop = np.array(new_output)
   # output_portiloop = np.convolve(output_portiloop, np.ones(w), 'full')[:len(output_portiloop)] / w
    #output_portiloop[output_portiloop >= 0.25] = 1

    output_portiloop = output_portiloop.astype(np.float)
    labels_portiloop = labels_portiloop.astype(np.float)
    tp = torch.Tensor(labels_portiloop * output_portiloop)
    tn = torch.Tensor((1 - labels_portiloop) * (1 - output_portiloop))
    fp = torch.Tensor((1 - labels_portiloop) * output_portiloop)
    fn = torch.Tensor((labels_portiloop * (1 - output_portiloop)))
    f1_test, precision_test, recall_test = get_metrics(tp, fp, fn)
    logging.debug(f"f1_test = {f1_test}")
    logging.debug(f"precision_test = {precision_test}")
    logging.debug(f"recall_test = {recall_test}")

    state = tp + fp * 2 + tn * 3 + fn * 4

    return f1_test, precision_test, recall_test, state, labels_portiloop, output_portiloop, test_loader


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    config_dict = get_config_dict(29, 0)
    smoothing_length = 3
    simulate(config_dict, smoothing_length,0)
