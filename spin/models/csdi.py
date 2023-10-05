import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.layers import PositionalEncoding
from tsl.nn.blocks.encoders.transformer import SpatioTemporalTransformerLayer
from tsl.nn.blocks.encoders.mlp import MLP

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=64, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


def positional_encoding(max_len, hidden_dim):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
    pe = torch.zeros(max_len, hidden_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def generate_positional_encoding(B, hidden_dim, K, L):
    pe = positional_encoding(L, hidden_dim)  # (L, hidden_dim)
    pe = pe.permute(1, 0)  # (hidden_dim, L)
    pe = pe.unsqueeze(0).unsqueeze(2)  # (1, hidden_dim, 1, L)
    pe = pe.expand(B, hidden_dim, K, L)  # (B, hidden_dim, K, L)
    return pe

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer



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


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=hidden_dim)

        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=hidden_dim)



    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x):

        B, channel, K, L = x.shape

        base_shape = x.shape
        y = x.reshape(B, channel, K * L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)

        x = y.reshape(base_shape)

        return x


class CsdiModel(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 covariate_dim=0,
                 nheads=1,
                 spatial_dim=36
                 ):


        super().__init__()

        self.hidden_dim = hidden_dim

        self.spatial_embedding_layer = SpatialEmbedding(spatial_dim, hidden_dim)

        self.pe = PositionalEncoding(hidden_dim)
        self.cond_projection = MLP(covariate_dim, hidden_dim, n_layers=2)
        self.input_projection = MLP(input_dim * 2, hidden_dim, n_layers=2)
        self.output_projection = MLP(input_size=hidden_dim,
                                    hidden_size=hidden_dim,
                                    output_size=input_dim,
                                    n_layers=2)

        self.st_transformer_layer = SpatioTemporalTransformerLayer(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            ff_size=hidden_dim,
            n_heads=nheads,
            activation='relu',
            causal=False,
            dropout=0.)

        self.mask_token = StaticGraphEmbedding(1, hidden_dim)

        self.diffusion_embedding =  DiffusionEmbedding(
            num_steps=50,
            embedding_dim=hidden_dim,
        )




    def forward(self, x, mask, noisy_data, diffusion_step, side_info=None, **kwargs):
        # x: [batches steps nodes features]
        x = x * mask + noisy_data * (1-mask)
        x = torch.cat([x, mask], dim=-1)
        x = self.input_projection(x)

        if side_info is not None:
            side_info = self.cond_projection(side_info)
            x = x + side_info
        x = self.pe(x)

        # space encoding
        B, L, K, C = x.shape
        spatial_emb = self.spatial_embedding_layer(B, L)
        spatial_emb = spatial_emb.permute(0, 3, 2, 1)  # (B, C, K, L)
        x = x + spatial_emb

        # diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        x = x + diffusion_emb.unsqueeze(1).unsqueeze(2)


        x = self.st_transformer_layer(x)
        x = self.output_projection(x)

        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--covariate_dim', type=int, default=0)
        parser.add_argument('--input_dim', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--spatial_dim', type=int, default=36)

        return parser