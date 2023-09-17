import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky

from .utils import positional_encoding

class DynamicST(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self, num_nodes, seq_len, seed=42):
        df, dist = self.load(num_nodes, seq_len, seed)
        temporal_encoding= positional_encoding(seq_len, 1, 4).squeeze(1)
        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, temporal_encoding=temporal_encoding))

    def load(self, num_nodes, seq_len, seed):
        rng = np.random.RandomState(seed)
        dist = np.arange(0, num_nodes)
        dist = np.abs(dist[:, np.newaxis] - dist[np.newaxis, :])

        var_space = 1 * np.exp(-dist / 1)
        L_space = cholesky(var_space, lower=True)

        y = np.zeros((seq_len, num_nodes))

        eta = rng.normal(0, 1, size=(seq_len, num_nodes)) * 0.2
        eta = np.einsum('ij,jk->ik', eta, L_space)



        y[0, :] = rng.normal(0, 1, size=(num_nodes))
        for t in range(1, seq_len):
            for s in range(num_nodes):
                for x in range(num_nodes):
                    weight = self.transition_kernel(s, x, dist, self.spatial_varying_theta(s))
                    y[t, s] += weight * y[t-1, x]

            y[t, :] = y[t, :] + eta[t, :]

        epsilon = rng.normal(0, 1, size=(seq_len, num_nodes)) * 0.1
        z = y + epsilon

        plt.figure()
        time_coords = np.arange(0, seq_len)
        plt.plot(time_coords, z)
        plt.show()

        df = pd.DataFrame(z)
        df.index = pd.to_datetime(df.index)

        return df, dist


    def spatial_varying_theta(self, s):
        theta = dict(gamma=0.2, l=5, offset=0)
        return theta

    def transition_kernel(self, s1, s2, dist, theta):
        gamma = theta['gamma']
        l = theta['l']
        offset = theta['offset']
        return gamma * np.exp(-(dist[s1, s2] - offset) ** 2 / l)


    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)







if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 5, 50
    dataset = DynamicST(num_nodes, seq_len)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




