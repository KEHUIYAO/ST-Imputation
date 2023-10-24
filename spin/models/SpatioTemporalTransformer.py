from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import TransformerLayer
from tsl.nn.layers import PositionalEncoding
from tsl.utils.parser_utils import ArgParser, str_to_bool
import torch
from tsl.nn.layers.norm import LayerNorm
from tsl.nn.base.attention import MultiHeadAttention
from tsl.nn.utils import utils
from functools import partial

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

WindowProcess = None
WindowProcessReverse = None



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        # x = shortcut + self.drop_path(x)

        # # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SpatioTemporalTransformerLayer(nn.Module):
    r"""
    A TransformerLayer which attend both the spatial and temporal dimensions by stacking two `MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): Whether to causally mask the attention scores (can be `True` only if `axis` is `steps`).
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='steps',
                                               causal=causal)

        # self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
        #                                       qdim=hidden_size,
        #                                       kdim=hidden_size,
        #                                       vdim=hidden_size,
        #                                       heads=n_heads,
        #                                       axis='nodes',
        #                                       causal=False)


        self.spatial_att = nn.ModuleList([SwinTransformerBlock(dim=hidden_size, input_resolution=(12, 12), num_heads=1, window_size=6, shift_size=0, mlp_ratio=1),
                                          SwinTransformerBlock(dim=hidden_size, input_resolution=(12, 12), num_heads=1, window_size=6, shift_size=3, mlp_ratio=1)
                                          ])

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)

        self.mlp = nn.Sequential(
            LayerNorm(hidden_size),
            nn.Linear(hidden_size, ff_size),
            utils.get_layer_activation(activation)(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """"""
        # x: [batch, steps, nodes, features]

        x = self.skip_conn(x) + self.dropout(self.temporal_att(self.norm1(x), attn_mask=mask)[0])


        # # reshape x to be [batch*steps, nodes, features]
        # B, L, K, C = x.shape
        # x = x.view(B*L, K, C)
        #
        # for layer in self.spatial_att:
        #     x =  x + self.dropout(layer(x))
        #
        # # reshape x back to be [batch, steps, nodes, features]
        # x = x.view(B, L, K, C)

        x = x + self.mlp(x)
        return x

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


class SpatioTemporalTransformerModel(nn.Module):
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
        super().__init__()

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

        h = self.pe(h)


        # space encoding
        B, L, K, C = h.shape
        spatial_emb = self.spatial_embedding_layer(B, L)
        spatial_emb = spatial_emb.permute(0, 3, 2, 1)  # (B, C, K, L)
        # h = h + spatial_emb

        out = []
        for encoder, mlp, layer_norm in zip(self.encoder, self.readout, self.layer_norm):
            h = encoder(h)
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
