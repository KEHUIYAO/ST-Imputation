import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
from .utils import positional_encoding




class GaussianProcess(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self, num_nodes, seq_len, seed=42):
        df, dist = self.load(num_nodes, seq_len, seed)
        temporal_encoding= positional_encoding(seq_len, 1, 4).squeeze(1)
        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, temporal_encoding=temporal_encoding))

    def matern_covariance(self, x1, x2, length_scale=1.0, nu=1.5, sigma=1.0):
        dist = np.linalg.norm(x1 - x2)
        if dist == 0:
            return sigma ** 2
        coeff1 = (2 ** (1 - nu)) / gamma(nu)
        coeff2 = (np.sqrt(2 * nu) * dist) / length_scale
        return sigma ** 2 * coeff1 * (coeff2 ** nu) * kv(nu, coeff2)


    def load(self, num_nodes, seq_len, seed):
        y_list = []
        cur = seq_len
        while cur > 0:
            cur_next = max(0, cur-200)
            cur_seq_len = cur - cur_next
            y, dist = self.load_small(num_nodes, cur_seq_len, seed)
            y_list.append(y)
            cur = cur_next

        y = np.concatenate(y_list, axis=0)
        time_coords = np.arange(0, seq_len)

        plt.figure()
        plt.plot(time_coords, y)
        plt.show()

        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index)

        return df, dist


    def load_small(self, num_nodes, seq_len, seed):

        rng = np.random.RandomState(seed)

        time_coords = np.arange(0, seq_len)
        space_coords = np.random.rand(num_nodes, 2)
        dist = cdist(space_coords, space_coords)

        # create the temporal covariance matrix
        length_scale = 5
        nu = 1.5
        sigma = 1.0
        var_temporal = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                var_temporal[i, j] = self.matern_covariance(time_coords[i], time_coords[j], length_scale, nu, sigma)

        # create the spatial covariance matrix
        length_scale = 0.1
        nu = 1.5
        sigma = 1.0
        var_spatial = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                var_spatial[i, j] = self.matern_covariance(space_coords[i, :], space_coords[j, :], length_scale, nu, sigma)

        L_spatial = cholesky(var_spatial + 1e-6 * np.eye(num_nodes), lower=True)
        L_temporal = cholesky(var_temporal + 1e-6 * np.eye(seq_len), lower=True)

        eta = rng.normal(0, 1, num_nodes * seq_len)
        L_spatial_temporal = np.kron(L_spatial, L_temporal)
        y = np.einsum('ij, j->i', L_spatial_temporal, eta)
        y = y.reshape(num_nodes, seq_len)
        y = y.T

        return y, dist



    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)




if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 5, 100
    dataset = GaussianProcess(num_nodes, seq_len)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




