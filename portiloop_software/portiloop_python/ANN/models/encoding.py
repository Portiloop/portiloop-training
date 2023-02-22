from abc import abstractmethod
import ftplib
import torch
from torch import nn
import math
from einops import rearrange
from enum import Enum
import torch.nn.functional as F


class EncodingTypes(Enum):
    POSITIONAL_ENCODING = 1
    ONE_HOT_ENCODING = 2
    NO_ENCODING = 3


def pad_one_hot(input):
    return F.pad(
        input=input,
        pad=(0, 0, 1, 0, 0, 0),
        mode='constant',
        value=0
    )

class PositionalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoding_type = config['encoding_type'] 
        self.pos_encoder = PositionalEncoding(config['d_model'], config['device'], config['dropout']) if self.encoding_type is EncodingTypes.POSITIONAL_ENCODING else None
        if self.encoding_type == EncodingTypes.ONE_HOT_ENCODING:
            self.generate_one_hot_vecs()

    def forward(self, x):
        if self.encoding_type == EncodingTypes.POSITIONAL_ENCODING:
            x = rearrange(x, 'b s e -> s b e')
            x = self.pos_encoder(x)
            x = rearrange(x, 's b e -> b s e')
        elif self.encoding_type == EncodingTypes.ONE_HOT_ENCODING:
            if x.size(0) == self.one_hot_tensor_train.size(0):
                x = torch.cat((x, self.one_hot_tensor_train[:, :x.size(1), :]), -1)
            elif x.size(0) == self.one_hot_tensor_val.size(0):
                x = torch.cat((x, self.one_hot_tensor_val[:, :x.size(1), :]), -1)
            elif x.size(0) == self.one_hot_tensor_test.size(0):
                x = torch.cat((x, self.one_hot_tensor_test[:, :x.size(1), :]), -1)
            else:
                raise ValueError("Missing batch size in one hot encoding.")
        elif self.encoding_type == EncodingTypes.NO_ENCODING:
            pass
        return x

    def generate_one_hot_vecs(self):
        one_hot_tensor_train = torch.diag(torch.ones(self.config['seq_len'])).unsqueeze(0).expand(\
            self.config['batch_size'], self.config['seq_len'], -1).to(self.config['device'])
        one_hot_tensor_val = torch.diag(torch.ones(self.config['seq_len'])).unsqueeze(0).expand(\
            self.config['batch_size_validation'], self.config['seq_len'], -1).to(self.config['device'])
        one_hot_tensor_test = torch.diag(torch.ones(self.config['seq_len'])).unsqueeze(0).expand(\
            self.config['batch_size_test'], self.config['seq_len'], -1).to(self.config['device'])
        self.one_hot_tensor_train = pad_one_hot(one_hot_tensor_train) 
        self.one_hot_tensor_val = pad_one_hot(one_hot_tensor_val)
        self.one_hot_tensor_test = pad_one_hot(one_hot_tensor_test)


class PositionalEncoding(nn.Module):
    def __init__(self, 
    d_model: int, 
    device: torch.device,
    dropout: float = 0.1, 
    max_len: int = 5000):
        """Positional Encoding Module. Adds positional encoding using cosine and sine waves to the input data

        Args:
            d_model (int): Desired size of the Encoding dimension
            dropout (float, optional): dropout value. Defaults to 0.1.
            max_len (int, optional): maximum length of sequence. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        
        Returns:
            tensor: returns a tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SineEncoding(nn.Module):
    def __init__(self, 
    in_features: int,
    out_features: int):
        """Method to perform Sine encoding

        Args:
            in_features (int): Number of features of input vector
            out_features (int): Number of desired output features
        """
        super(SineEncoding, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return SineEncoding.t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

    @abstractmethod
    def t2v(tau, f, w, b, w0, b0, arg=None):
        if arg:
            v1 = f(torch.matmul(tau, w) + b, arg)
        else:
            v1 = f(torch.matmul(tau, w) + b)
        v2 = torch.matmul(tau, w0) + b0
        return torch.cat([v1, v2], 1)
