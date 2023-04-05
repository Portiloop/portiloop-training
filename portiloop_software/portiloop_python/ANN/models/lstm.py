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

from portiloop_software.portiloop_python.ANN.models.model_blocks import (
    AttentionLayer, FullAttention, TransformerEncoderLayer)
from portiloop_software.portiloop_python.ANN.utils import get_configs

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
            nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)

        output_cnn_size = int(nb_channel * nb_out)

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
        out_features = 1

        self.attention_layer = TransformerEncoderLayer(
            attention=AttentionLayer(
                FullAttention(),
                fc_features,
                n_heads
            ),
            d_model=fc_features,
            dropout=dropout_p,
            norm_layer=nn.LayerNorm(fc_features)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, fc_features))

        out_cnn_size_after_rnn = 0

        if self.after == "attention":
            in_fc = fc_features * 2
        elif self.after == "cnn":
            in_fc = fc_features + out_cnn_size_after_rnn
        else:
            in_fc = fc_features

        self.fc = nn.Linear(in_features=in_fc,
                            out_features=out_features)  # probability of being a spindle


    def forward(self, x, h, past_x=None, max_value=np.inf):
        # x: input data (batch_size, sequence_len, features)
        # h: hidden state of the GRU (nb_rnn_layers, batch_size, hidden_size)
        # past_x: accumulated past embeddings (batch_size, any_seq_len, features)
        # max_value (optional) : print the maximal value reach during inference (used to verify if the FPGA implementation precision is enough)
        (batch_size, sequence_len, features) = x.shape

        x = x.view(-1, 1, features)
        x = self.cnn(x)
        x = self.cnn2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(batch_size, sequence_len, -1)
        x, h = self.gru(x, h)

        out_gru = x[:, -1, :]  # output size: 1

        if self.after == "attention":
            
            # Use the accumulated past embeddings for attention (for validation)
            if past_x is not None:
                x = torch.cat((past_x, x), dim=1)

            # Append the cls token to the output of the GRU
            cls = self.cls.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

            out_attention = self.attention_layer(x)

            # Concatenate the output of the GRU and the output of the attention layer
            out = torch.cat((out_gru, out_attention[:, 0, :]), dim=1)
        elif self.after == "cnn":
            out = out_gru
            # Add CNN experiment
        else:
            out = out_gru

        out = self.fc(out)

        x = torch.sigmoid(out)

        return x, h, out_gru.unsqueeze(1)


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
    net.load_state_dict(checkpoint['model_state_dict'])
    return net    

if __name__ ==  "__main__":

    
    config = get_configs("Test", True, 42)
    config['nb_conv_layers'] = 4
    config['hidden_size'] = 64
    config['nb_rnn_layers'] = 4

    model = PortiloopNetwork(config)
    summary(model)

    window_size = int(config['window_size_s'] * config['fe'])
    x = torch.randn(config['batch_size'], config['seq_len'], window_size)
    h = torch.zeros(config['nb_rnn_layers'], config['batch_size'], config['hidden_size'])
    start = time.time()
    res_x, res_h, _ = model(x, h)
    end = time.time()
    print("Time taken: ", end - start)
