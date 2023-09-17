import numpy as np
import pandas as pd
import torch
import itertools
from torch import nn


class MeanModel(nn.Module):
    def forward(self,
                x,
                mask,
                **kwargs):

        # x: [batch, steps, nodes, channels]
        B = x.shape[0]
        for b in range(B):
            x[b] = torch.mean(x[b, mask[b].bool()])

        x[torch.isnan(x)] = 0

        return x

    @staticmethod
    def add_model_specific_args(parser):
        return parser