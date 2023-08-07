from torch import tensor
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import repeat
from math import sqrt
from copy import deepcopy

from portiloop_ml.portiloop_python.ANN.models.encoding import PositionalEncoder
from portiloop_ml.portiloop_python.ANN.models.masking import FullMask, LengthMask


class Transformer(nn.Module):
    def __init__(
        self,
        config
    ):
        """Prediction module to build models used for pretraining on EEG data.

        Args:
            d_model (int): Dimension of the hidden layers
            n_heads (int): Number of heads for each encoder layer
            dim_hidden (int): Dimension of the Feedforward layer for each TRansformer Encoder Layer
            n_layers (int): Number of Transformer encoder layers
            prediction_len (int): Length of the unseen predicted sequence
            seq_len (int): Length of the seen recreated sequence for Autoencoder  
            latent_dim (int): Dimension of the hidden latent dimension fro autoencoder
            num_channels_deconv (int): Number of Channels in the deconvolution layers
            num_layers_deconv (int): Number of layers in the deconvolution
            dropout (float, optional): Defaults to 0.5.
        """
        super().__init__()

        self.config = config

        # Initialize the Transformer Encoder and Decoder if necessary
        d_model = config['d_model']
        n_heads = config['n_heads']
        dim_ff = config['dim_ff']
        n_layers = config['n_layers']
        device = config['device']
        dropout = config['dropout']
        q_dim = config['q_dim']
        v_dim = config['v_dim']

        self.transformer_encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        FullAttention(),
                        d_model,
                        n_heads,
                        d_keys=q_dim,
                        d_values=v_dim,
                    ),
                    d_model,
                    dim_ff,
                    dropout,
                    'gelu',
                    (nn.LayerNorm(d_model)
                     if config['normalization'] else None)
                )
                for _ in range(n_layers)
            ],
            (nn.LayerNorm(d_model) if config['final_norm'] else None),
        )
        if config['full_transformer']:
            self.transformer_decoder = TransformerDecoder(
                [
                    TransformerDecoderLayer(
                        AttentionLayer(
                            FullAttention(),
                            d_model,
                            n_heads,
                            d_keys=q_dim,
                            d_values=v_dim,
                        ),
                        AttentionLayer(
                            FullAttention(),
                            d_model,
                            n_heads,
                            d_keys=q_dim,
                            d_values=v_dim,
                        ),
                        d_model,
                        dim_ff,
                        dropout,
                        'gelu',
                        (nn.LayerNorm(d_model)
                         if config['normalization'] else None)
                    )
                    for _ in range(n_layers)
                ],
                (nn.LayerNorm(d_model) if config['final_norm'] else None),
            )
        else:
            self.transformer_decoder = None

        # Initialize CLS token for classification
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config['embedding_size']))

        # Positional encoder than can do either One hot encoding of position or positional encoder as intended in Transformer
        self.positional_encoder = PositionalEncoder(config)

    def forward(self, x: tensor, history: tensor):
        """_summary_

        Args:
            x (tensor): Input Tensor of dimension [batch_size, seq_len]

        Returns:
            x_pred (tensor): Output tensor of Dimension [batch_size, prediction_len] after going through the Autoencoder model
            x_rec (tensor): Output tensor of Dimension [batch_size, seq_len] after going through the Autoencoder model
        """

        # Add the cls token at the beggining of the sequence
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding to our model
        x = self.positional_encoder(x)

        # Go through feature extractor
        x = self.transformer_encoder(x)

        # Go through transformer decoder with history as x and memory as input
        if self.config['full_transformer']:
            history = history.unsqueeze(-1).expand(history.size(0),
                                                   history.size(1), self.config['embedding_size'])
            history = self.positional_encoder(history)
            x = self.transformer_decoder(history, x)

        return x


class TransformerEncoder(nn.Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.

        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    """TransformerDecoder is little more than a sequence of transformer decoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ----------
        layers: list, TransformerDecoderLayer instances or instances that
                implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # Apply all the transformer decoders
        for layer in self.layers:
            x = layer(x, memory, x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask,
                      memory_length_mask=memory_length_mask)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", norm_layer=None):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = deepcopy(norm_layer)
        self.norm2 = deepcopy(norm_layer)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        ))

        # Run the fully connected part of the layer
        if self.norm1 is not None:
            x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        if self.norm2 is not None:
            res = self.norm2(x+y)
        else:
            res = x+y

        return res


class TransformerDecoderLayer(nn.Module):
    """The decoder layer from "Attention Is All You Need".

    Similar to the encoder layer, this layer implements the decoder that
    PyTorch implements but can be used with any attention implementation
    because it receives the attention layers as constructor arguments.

    Arguments
    ---------
        self_attention: The attention implementation to use for self attention
                        given as a nn.Module
        cross_attention: The attention implementation to use for cross
                         attention given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", normalization=None):
        super(TransformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = deepcopy(normalization)
        self.norm2 = deepcopy(normalization)
        self.norm3 = deepcopy(normalization)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None):
        """Apply the transformer decoder to the input x using the memory
        `memory`.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E should be the same as
               the d_model passed in the constructor.
            memory: The memory features of shape (N, L', E) where N is the
                    batch size, L' is the memory's sequence length (padded) and
                    E should be the same as the d_model.
            x_mask: An implementation of fast_transformers.masking.BaseMask
                    that encodes where each element of x can attend to in x.
                    Namely the self attention mask.
            x_length_mask: An implementation of a BaseMask that encodes how
                           many elements each sequence in the batch consists
                           of.
            memory_mask: An implementation of BaseMask that encodes where each
                         element of x can attend to in the memory. Namely the
                         cross attention mask.
            memory_length_mask: An implementation of a BaseMask that encodes how
                                many elements each memory sequence in the batch
                                consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = memory.shape[1]
        x_mask = x_mask or FullMask(L, device=x.device)
        x_length_mask = x_length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))
        memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
        memory_length_mask = memory_length_mask or \
            LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
        if self.norm1 is not None:
            x = self.norm1(x)

        # Secondly apply the cross attention and add it to the previous output
        x = x + self.dropout(self.cross_attention(
            x, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask
        ))

        # Finally run the fully connected part of the layer
        if self.norm2 is not None:
            y = x = self.norm2(x)
        else:
            y = x
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        if self.norm3 is not None:
            res = self.norm3(x+y)
        else:
            res = x + y

        return res


class AttentionLayer(nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class FullAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Make sure that what we return is contiguous
        return V.contiguous()


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Flattens the input sequence to a 2D tensor
        self.flattener = nn.Flatten()

        self.shallownet = nn.Sequential(
            nn.Conv1d(1, config['d_model'], config['conv_ker_size'],
                      stride=config['conv_ker_size']//2),
            nn.BatchNorm1d(config['d_model']),
            nn.ELU(),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.AvgPool1d(config['pool_ker_size'],
                         stride=config['pool_ker_size']//10),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        # x -> (batch_size, seq_len, window_size)
        # x -> (batch_size, 1, seq_len*window_size)
        x = self.flattener(x).unsqueeze(1)
        x = self.shallownet(x)  # x -> (batch_size, num_channels, seq_len)
        # Invert the last two dimensions
        x = x.permute(0, 2, 1)  # x -> (batch_size, seq_len, num_channels)
        return x


# get cnn encoder
def get_cnn_embedder(config):
    """
    This function generates the CNN encoder and returns the sequence length for the transformer encoder
    """
    cnn_embedder = PatchEmbedding(config)
    test = torch.rand(config['batch_size'],
                      config['seq_len'], config['window_size'])
    seq_len = cnn_embedder(test).size(1)
    return cnn_embedder, seq_len


def build_cnn_layers(config):
    # Checking if verification of CNN layers' dimensions has been previously done
    model = nn.Sequential()

    # Generating the CNN layers
    in_channels = config['cnn_in_channels']
    for _ in range(config['cnn_num_layers']):
        out_channels = config['cnn_channels']
        model.append(ConvPoolModule(
            in_channels=in_channels,
            out_channel=out_channels,
            kernel_conv=config['cnn_kernel_size'],
            stride_conv=config['cnn_stride_conv'],
            conv_padding=config['cnn_padding'],
            dilation_conv=config['cnn_dilation'],
            kernel_pool=config['pool_kernel_size'],
            stride_pool=config['pool_stride_conv'],
            pool_padding=config['pool_padding'],
            dilation_pool=config['pool_dilation'],
            dropout_p=config['dropout']
        ))
        in_channels = out_channels

    return model


def build_encoder_module(config):
    # Checking if verification of CNN layers' dimensions has been previously done
    model = nn.Sequential()
    assert config['cnn_linear_size'] > 0, "Error in config, make sure to verify CNN sizes before generating model."

    # Generating the CNN layers
    in_channels = config['cnn_in_channels']
    for _ in range(config['cnn_num_layers']):
        out_channels = config['cnn_channels_multiplier'] * in_channels
        model.append(ConvPoolModule(
            in_channels=in_channels,
            out_channel=out_channels,
            kernel_conv=config['cnn_kernel_size'],
            stride_conv=config['cnn_stride_conv'],
            conv_padding=config['cnn_padding'],
            dilation_conv=config['cnn_dilation'],
            kernel_pool=config['pool_kernel_size'],
            stride_pool=config['pool_stride_conv'],
            pool_padding=config['pool_padding'],
            dilation_pool=config['pool_dilation'],
            dropout_p=config['dropout']
        ))
        in_channels = out_channels

    # Generating Linear to project CNN output onto d_model
    model.append(nn.Linear(config['cnn_linear_size'],
                 config['embedding_size']//out_channels))
    model.append(nn.Flatten())
    model.append(nn.ReLU())
    return model


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
