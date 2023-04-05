import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvAtt(nn.Module):
    def __init__(self, embed_dim_in, embed_dim_out, kernel_size, num_heads, dropout=0.1):
        super(ConvAtt, self).__init__()
        
        self.conv1d = nn.Conv1d(embed_dim_in, embed_dim_out, kernel_size, padding='same')
        self.layer_norm1 = nn.LayerNorm(embed_dim_out)
        self.self_att = nn.MultiheadAttention(embed_dim_out, num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim_out)
        self.dropout_val = dropout
        
    def forward(self, x):
        # x.shape: (seq_len, embed_dim)
        
        # Convolution pass
        x = self.conv1d(x)
        x = einops.rearrange(x, 'b e s -> b s e')  # (batch_size, seq_len, embed_dim)
        x = F.relu(self.layer_norm1(x))
        
        # Attention pass
        x = self.layer_norm2(x)
        x, _ = self.self_att(x, x, x)  # query, key, value all come from x
        
        # Residual connection
        x = x + F.dropout(x, p=self.dropout_val, training=self.training)
        x = einops.rearrange(x, 'b s e -> b e s')  # (batch_size, embed_dim, seq_len)
        return x
    
class PortiConvAtt(nn.Module):
    def __init__(self) -> None:
        super(PortiConvAtt, self).__init__()
        self.conv_att1 = ConvAtt(1, 32, 4, 4)
        self.conv_att2 = ConvAtt(32, 64, 4, 4)
        self.conv_att3 = ConvAtt(64, 128, 4, 4)

        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)

        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_att1(x)
        x = self.conv_att2(x)
        x = self.conv_att3(x)
        
        x = self.avg_pool1(x)

        x = x.squeeze(-1)

        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1, stride=1, padding='same')
        
    def forward(self, x):
        x = self.conv(x)
        return x
    

class PortiResNet(nn.Module):
    def __init__(self, depth, hidden_rnn, num_layers_rnn):
        super(PortiResNet, self).__init__()
        self.convs = nn.Sequential(ResNetBlock(1, 4))
        multiplier = 2
        for i in range(depth):
            self.convs.append(ResNetBlock(pow(2, multiplier+i), pow(2, multiplier+i+1), downsample=Downsample(pow(2, multiplier+i))))
        
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)

        self.rnn = nn.GRU(
            input_size=pow(2, multiplier + depth),
            hidden_size=hidden_rnn,
            num_layers=num_layers_rnn,
            dropout=0,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_rnn, 1)

    def forward(self, x, h):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = einops.rearrange(x, 'b s w -> (b s) 1 w')
        for conv in self.convs:
            x = conv(x)
        
        x = self.avg_pool1(x)
        x = einops.rearrange(x, '(b s) e 1 -> b s e', b=batch_size, s=seq_len)  # (batch_size, seq_len, embed_dim)
        x, h = self.rnn(x, h)

        out_gru = x[:, -1, :]

        x = self.linear(out_gru)
        x = torch.sigmoid(x)
        return x, h, None


if __name__ == "__main__":
    batch_size = 64
    seq_len = 50
    embed_dim_in = 1
    embed_dim_out = 128
    kernel_size = 4
    num_heads = 4
    # model = ConvAtt(embed_dim_in, embed_dim_out, kernel_size, num_heads)
    depth = 5
    hidden_size = 32
    num_layers = 4
    model = PortiResNet(depth, hidden_size, num_layers)
    summary(model)

    window_size = 54
    # Test forward pass
    x = torch.randn(batch_size, seq_len, window_size)
    h = torch.zeros(num_layers, batch_size, hidden_size)
    y = model(x, h)
    