from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import SpatioTemporalTransformerLayer, \
    TransformerLayer
from tsl.nn.layers import PositionalEncoding
from tsl.utils.parser_utils import ArgParser, str_to_bool
import torch
from tsl.nn.layers.norm import LayerNorm
class SpatialEmbedding(nn.Module):
    def __init__(self, K, channel):
        super(SpatialEmbedding, self).__init__()

        self.K = K
        self.channel = channel

        # Trainable embedding layer
        self.embedding = nn.Embedding(K, channel)

    def forward(self, B, L):
        # Generate embeddings for [0, 1, ..., K-1]
        x = torch.arange(self.K).to(self.embedding.weight.device)
        embed = self.embedding(x)  # (K, channel)
        embed = embed.permute(1, 0)  # (channel, K)
        embed = embed.unsqueeze(0).unsqueeze(3)  # (1, channel, K, 1)
        embed = embed.expand(B, self.channel, self.K, L)  # (B, channel, K, L)
        return embed


class TransformerModel(nn.Module):
    r"""Spatiotemporal Transformer for multivariate time series imputation.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        u_size (int): Dimension of the exogenous variables.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations.
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 ff_size: int,
                 u_size: int,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 condition_on_u: bool = True,
                 axis: str = 'both',
                 activation: str = 'elu',
                 spatial_dim: int = 36):
        super(TransformerModel, self).__init__()

        self.condition_on_u = condition_on_u
        if condition_on_u:
            self.u_enc = MLP(u_size, hidden_size, n_layers=2)
        self.h_enc = MLP(input_size, hidden_size, n_layers=2)

        self.mask_token = StaticGraphEmbedding(1, hidden_size)

        self.pe = PositionalEncoding(hidden_size)

        self.spatial_embedding_layer = SpatialEmbedding(spatial_dim, hidden_size)

        kwargs = dict(input_size=hidden_size,
                      hidden_size=hidden_size,
                      ff_size=ff_size,
                      n_heads=n_heads,
                      activation=activation,
                      causal=False,
                      dropout=dropout)

        if axis in ['steps', 'nodes']:
            transformer_layer = TransformerLayer
            kwargs['axis'] = axis
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(transformer_layer(**kwargs))
            self.layer_norm.append(LayerNorm(hidden_size))
            self.readout.append(MLP(input_size=hidden_size,
                                    hidden_size=ff_size,
                                    output_size=output_size,
                                    n_layers=2,
                                    dropout=dropout))

    def forward(self, x, mask, u=None, **kwargs):
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        x = x * mask

        h = self.h_enc(x)
        h = mask * h + (1 - mask) * self.mask_token()

        if self.condition_on_u and u is not None:
            h = h + self.u_enc(u)

        #h = self.pe(h)


        # space encoding
        B, L, K, C = h.shape
        spatial_emb = self.spatial_embedding_layer(B, L)
        spatial_emb = spatial_emb.permute(0, 3, 2, 1)  # (B, C, K, L)
        #h = h + spatial_emb

        out = []
        for encoder, mlp, layer_norm in zip(self.encoder, self.readout, self.layer_norm):
            #h = encoder(h)
            h = layer_norm(h)
            out.append(mlp(h))

        x_hat = out.pop(-1)
        return x_hat, out

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--n-layers', type=int, default=4, tunable=True,
                        options=[1, 2, 3, 4])
        parser.opt_list('--n-heads', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--dropout', type=float, default=0., tunable=True,
                        options=[0., 0.1, 0.25, 0.5])
        parser.add_argument('--condition-on-u', type=str_to_bool, nargs='?',
                            const=True, default=True)
        parser.opt_list('--axis', type=str, default='both', tunable=True,
                        options=['steps', 'both'])

        parser.add_argument('--input_size', type=int, default=1)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--ff_size', type=int, default=64)
        parser.add_argument('--u_size', type=int, default=0)
        parser.add_argument('--spatial_dim', type=int, default=36)


        return parser
