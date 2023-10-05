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


class SpatioTemporalTransformerModel(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 covariate_dim=0,
                 nheads=1,
                 nlayers=1,
                 spatial_dim=36
                 ):


        super().__init__()

        self.hidden_dim = hidden_dim

        self.spatial_embedding_layer = SpatialEmbedding(spatial_dim, hidden_dim)

        self.input_projection = Conv1d_with_init(input_dim, hidden_dim, 1)
        self.output_projection1 = Conv1d_with_init(hidden_dim, hidden_dim, 1)
        self.output_projection2 = Conv1d_with_init(hidden_dim, input_dim, 1)
        self.cond_projection = Conv1d_with_init(covariate_dim + input_dim, hidden_dim, 1)
        self.pe = PositionalEncoding(hidden_dim)

        self.st_transformer_layer = SpatioTemporalTransformerLayer(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            ff_size=hidden_dim,
            n_heads=nheads,
            activation='relu',
            causal=False,
            dropout=0.)



        # self.residual_layers = nn.ModuleList(
        #     [
        #         ResidualBlock(
        #             hidden_dim=hidden_dim,
        #             nheads=nheads
        #         )
        #         for _ in range(nlayers)
        #     ]
        # )

        self.mask_token = StaticGraphEmbedding(1, hidden_dim)


    def get_side_info(self, cond_mask, side_info):
        if side_info is not None:
            cond_info = torch.cat([cond_mask, side_info], dim=3)  # (B,L,K,cond_dim+input_dim)
        else:
            cond_info = cond_mask.float()


        cond_info = cond_info.permute(0, 3, 2, 1)  # (B,cond_dim,K,L)
        return cond_info

    def forward(self, x, mask, side_info=None, **kwargs):
        hidden_dim = self.hidden_dim
        x = mask * x
        x = x.permute(0, 3, 2, 1)



        B, inputdim, K, L = x.shape


        cond_info = self.get_side_info(mask, side_info)


        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, hidden_dim, K, L)





        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,channel,K*L)
        cond_info = F.relu(cond_info)
        cond_info = cond_info.reshape(B, hidden_dim, K, L)  # (B,channel,K,L)

        x = x + cond_info


        # temporal encoding
        x = x.permute(0, 3, 2, 1)  # (B,L,K,hidden_dim)

        x = self.pe(x)





        # # space encoding
        # spatial_emb = self.spatial_embedding_layer(B, L)
        # x = x + spatial_emb


        # for layer in self.residual_layers:
        #     x = layer(x)

        x = self.st_transformer_layer(x)

        x = x.permute(0, 3, 2, 1)  # (B,hidden_dim,K,L)

        x = x.reshape(B, hidden_dim, K * L)
        x = self.output_projection1(x) # (B,hidden_dim,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,input_dim,K*L)
        x = x.reshape(B, -1, K, L)  # (B,input_dim,K,L)
        x = x.permute(0, 3, 2, 1)  # (B, L, K ,input_dim)
        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--covariate_dim', type=int, default=0)
        parser.add_argument('--input_dim', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--spatial_dim', type=int, default=36)

        return parser