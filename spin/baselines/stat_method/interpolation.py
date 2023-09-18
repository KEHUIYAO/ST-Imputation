import numpy as np
import pandas as pd
import torch
import itertools
from torch import nn


class InterpolationModel(nn.Module):
    def forward(self,
                x,
                mask, **kwargs):

        # x: [batch, steps, nodes, channels]
        device = x.device
        x = x.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        x[mask==0] = np.nan
        B = x.shape[0]
        K = x.shape[2]
        C = x.shape[3]
        for b in range(B):
            for k in range(K):
                for c in range(C):
                    x[b, :, k, c] = pd.Series(x[b, :, k, c]).interpolate(method='linear', limit_direction='both').values
        x[np.isnan(x)] = 0
        x = torch.from_numpy(x).float().to(device)
        return x

    @staticmethod
    def add_model_specific_args(parser):
        return parser