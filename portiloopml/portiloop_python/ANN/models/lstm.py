import copy
import logging
from math import floor
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

from portiloopml.portiloop_python.ANN.models.model_blocks import (
    AttentionLayer, FullAttention, TransformerEncoderLayer)
from portiloopml.portiloop_python.ANN.utils import get_configs

ABLATION = 0


class ConvPoolModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 kernel_conv,
                 stride_conv,
                 conv_padding,
                 dilation_conv,
                 kernel_pool,
                 stride_pool,
                 pool_padding,
                 dilation_pool,
                 dropout_p):
        super(ConvPoolModule, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_conv,
                              stride=stride_conv,
                              padding=conv_padding,
                              dilation=dilation_conv)
        self.pool = nn.MaxPool1d(kernel_size=kernel_pool,
                                 stride=stride_pool,
                                 padding=pool_padding,
                                 dilation=dilation_pool)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return self.dropout(x)


class FcModule(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 dropout_p):
        super(FcModule, self).__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.dropout(x)


class PortiloopNetwork(nn.Module):
    def __init__(self, c_dict):
        super(PortiloopNetwork, self).__init__()

        RNN = c_dict["RNN"]
        stride_pool = c_dict["stride_pool"]
        stride_conv = c_dict["stride_conv"]
        kernel_conv = c_dict["kernel_conv"]
        kernel_pool = c_dict["kernel_pool"]
        nb_channel = c_dict["nb_channel"]
        in_channels = c_dict["in_channels"]
        hidden_size = c_dict["hidden_size"]
        window_size_s = c_dict["window_size_s"]
        dropout_p = c_dict["dropout"]
        dilation_conv = c_dict["dilation_conv"]
        dilation_pool = c_dict["dilation_pool"]
        fe = c_dict["fe"]
        nb_conv_layers = c_dict["nb_conv_layers"]
        nb_rnn_layers = c_dict["nb_rnn_layers"]
        first_layer_dropout = c_dict["first_layer_dropout"]
        self.envelope_input = c_dict["envelope_input"]
        self.power_features_input = c_dict["power_features_input"]
        self.classification = c_dict["classification"]
        n_heads = c_dict["n_heads"]
        self.after = c_dict['after_rnn']

        conv_padding = 0  # int(kernel_conv // 2)
        pool_padding = 0  # int(kernel_pool // 2)
        window_size = int(window_size_s * fe)
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding,
                             dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding,
                             dilation_pool, kernel_pool, stride_pool)

        output_cnn_size = int(nb_channel * nb_out)
        print(output_cnn_size)

        self.RNN = RNN

        self.cnn = ConvPoolModule(in_channels=1,
                                  out_channel=nb_channel,
                                  kernel_conv=kernel_conv,
                                  stride_conv=stride_conv,
                                  conv_padding=conv_padding,
                                  dilation_conv=dilation_conv,
                                  kernel_pool=kernel_pool,
                                  stride_pool=stride_pool,
                                  pool_padding=pool_padding,
                                  dilation_pool=dilation_pool,
                                  dropout_p=dropout_p if first_layer_dropout else 0)

        self.cnn2 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
                                                   out_channel=nb_channel,
                                                   kernel_conv=kernel_conv,
                                                   stride_conv=stride_conv,
                                                   conv_padding=conv_padding,
                                                   dilation_conv=dilation_conv,
                                                   kernel_pool=kernel_pool,
                                                   stride_pool=stride_pool,
                                                   pool_padding=pool_padding,
                                                   dilation_pool=dilation_pool,
                                                   dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))
        self.gru = nn.GRU(input_size=output_cnn_size,
                          hidden_size=hidden_size,
                          num_layers=nb_rnn_layers,
                          dropout=0,
                          batch_first=True)

        fc_features = 0
        fc_features += hidden_size
        out_features = c_dict['out_features']

        in_fc = fc_features
        self.hidden_fc = nn.Linear(in_features=fc_features,
                                    out_features=fc_features)

        self.fc_spindles = nn.Linear(in_features=in_fc,
                            out_features=out_features)  # probability of being a spindle
        
        self.fc_sleep_stage = nn.Linear(in_features=in_fc,
                            out_features=5)


    def forward(self, x, h, past_x=None, max_value=np.inf):
        # x: input data (batch_size, sequence_len, features)
        # h: hidden state of the GRU (nb_rnn_layers, batch_size, hidden_size)
        # past_x: accumulated past embeddings (batch_size, any_seq_len, features)
        # max_value (optional) : print the maximal value reach during inference (used to verify if the FPGA implementation precision is enough)
        (batch_size, sequence_len, in_channels, features) = x.shape

        x = x.view(-1, in_channels, features)
        x = self.cnn(x)
        x = self.cnn2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(batch_size, sequence_len, -1)
        x, h = self.gru(x, h)

        out_gru = x[:, -1, :]  # output size: 1

        out = self.hidden_fc(out_gru)
        out = torch.relu(out)
      
        out_spindles = self.fc_spindles(out)
        out_sleep_stage = self.fc_sleep_stage(out)

        # Returns:
        #   - the spindle classifier output (Shape (batch_size, 1))
        #   - the sleep stage classifier output (Shape (batch_size, 5))
        #   - the hidden state(s) of the GRU(s) (Shape (nb_rnn_layers, batch_size, hidden_size))
        #   - the embedding of the sequence (Shape (batch_size, hidden_size)) 
        return out_spindles, out_sleep_stage, h, out

def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def get_trained_model(config_dict, model_path):
    experiment_name = config_dict['experiment_name']
    device_inference = config_dict["device_inference"]
    classification = config_dict["classification"]
    if device_inference.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable"
    net = PortiloopNetwork(config_dict).to(device=device_inference)
    if not device_inference.startswith("cuda"):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return net


class WaveletCNN(nn.Module):
    def __init__(self, device, kernel_size=1, channels=1, bias=True, padding='same'):
        super().__init__()
        self.w = nn.Parameter(torch.ones(channels, 1) * 5, requires_grad=True)
        self.s = nn.Parameter(torch.ones(channels, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            channels), requires_grad=True) if bias else None
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.device = device

    def forward(self, x):
        kernels = self.get_kernels()
        result = F.conv1d(x, kernels, self.bias, padding=self.padding)
        return result

    def get_kernels(self, complete=True):
        """
        Get morlet wave
        """
        spaces = torch.linspace(-1, 1, self.kernel_size).unsqueeze(
            0).expand(self.channels, self.kernel_size).to(self.device)
        scales = (self.s * 2 * torch.pi)
        spaces = spaces * scales
        output = torch.cos(self.w * spaces)

        if complete:
            output -= torch.exp(-0.5 * (self.w**2))

        output *= torch.exp(-0.5 * (spaces**2)) * torch.pi**(-0.25)

        return output.unsqueeze(1)


if __name__ == "__main__":

    config = get_configs("Test", True, 42)
    # config['nb_conv_layers'] = 4
    # config['hidden_size'] = 64
    # config['nb_rnn_layers'] = 4
    config['hidden_size'] = 64
    config['after_rnn'] = 'hidden'

    model = PortiloopNetwork(config).to(config["device_train"])
    # summary(model)

    window_size = int(config['window_size_s'] * config['fe'])
    x = torch.randn(config['batch_size'], config['seq_len'],
                    1, window_size).to(config['device_train'])
    h = torch.zeros(config['nb_rnn_layers'], config['batch_size'],
                    config['hidden_size']).to(config['device_train'])
    start = time.time()
    res_x, res_h, _ = model(x, h)
    end = time.time()
    print("Time taken: ", end - start)
