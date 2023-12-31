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

    def __init__(self, mode='train'):
        self.original_data = {}

        df, dist, st_coords, eval_mask = self.load(mode=mode)


        super().__init__(dataframe=df,
                         similarity_score="distance",
                         attributes=dict(dist=dist,
                          st_coords=st_coords))

        self.set_eval_mask(eval_mask)


    def load(self, mode):

        with np.load(os.path.join(current_dir, 'hmnist_random.npz')) as data:

            if mode == 'train':
                y = data['x_train_full']
                eval_mask = data['m_train_miss']
            elif mode == 'test':
                y = data['x_test_full']
                eval_mask = data['m_test_miss']




        # reshape
        y = y.reshape((y.shape[0]*y.shape[1], y.shape[2]))
        eval_mask = eval_mask.reshape((eval_mask.shape[0]*eval_mask.shape[1], eval_mask.shape[2]))



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


        return y, dist, st_coords, eval_mask

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)




if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values
    dataset = HealingMnist('test')
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
