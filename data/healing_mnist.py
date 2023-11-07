import pandas as pd
import torch

from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin

from tsl.ops.similarities import gaussian_kernel

from tsl.data.datamodule.splitters import Splitter
import matplotlib.pyplot as plt
import numpy as np


from scipy.spatial.distance import cdist
import os
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))


class HealingMnist(PandasDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self):
        self.original_data = {}
        df, dist, st_coords = self.load()


        super().__init__(dataframe=df,
                         similarity_score="distance",
                         attributes=dict(dist=dist,
                          st_coords=st_coords))


    def load(self):

        with np.load(os.path.join(current_dir, 'hmnist_mnar.npz')) as data:
            x_train_full = data['x_train_full']

            y = x_train_full[1:1000, :, :]

        # reshape
        y = y.reshape((y.shape[0]*y.shape[1], y.shape[2]))


        rows, cols = y.shape


        self.original_data['y'] = y

        y = pd.DataFrame(y)

        y.index = pd.to_datetime(y.index)



        # spatiotemporal coords
        space_coords, time_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        st_coords = np.stack([space_coords, time_coords], axis=-1)




        dist = []
        for j in range(28):
            for i in range(28):
                dist.append([i, j])
        dist = np.array(dist)
        dist = cdist(dist, dist)


        return y, dist, st_coords

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)


    def get_splitter(self, method=None, **kwargs):
        return SoilMoistureSplitter(kwargs.get('val_len'), kwargs.get('test_len'))


# class SoilMoistureSplitter(FixedIndicesSplitter):
#     def __init__(self):
#         # train_idxs = []
#         # valid_idxs = []
#         # for i in range(36):
#         #     for j in range(365):
#         #         train_idxs.append(i*365 + j)
#         #
#         #     start = 10
#         #     end = 50
#         #
#         #     for j in range(start, end):
#         #         valid_idxs.append(i*365 + j)
#         #
#         #
#         # test_idxs = []
#         # for i in range(36):
#         #     for j in range(365):
#         #         test_idxs.append(i*365 + 365 + j)
#
#         train_idxs = [0, 1, 2]
#         valid_idxs = [3, 4, 5]
#         test_idxs = [6, 7, 8]
#
#         super().__init__(train_idxs, valid_idxs, test_idxs)


class SoilMoistureSplitter(Splitter):

    def __init__(self, val_len: int = None, test_len: int = None):
        super().__init__()
        self._val_len = val_len
        self._test_len = test_len

    def fit(self, dataset):
        idx = np.arange(len(dataset))
        val_len, test_len = self._val_len, self._test_len
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))

        # randomly split idx into train, val, test
        np.random.shuffle(idx)
        val_start = len(idx) - val_len - test_len
        test_start = len(idx) - test_len


        self.set_indices(idx[:val_start - dataset.samples_offset],
                         idx[val_start:test_start - dataset.samples_offset],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.2)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values
    dataset = HealingMnist()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
