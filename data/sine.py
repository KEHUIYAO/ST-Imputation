import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky





class Sine(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self, num_nodes, seq_len, seed=42):
        self.original_data = {}
        df, dist, st_coords = self.load(num_nodes, seq_len, seed)
        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, st_coords=st_coords))



    def load(self, num_nodes, seq_len, seed):

        rng = np.random.RandomState(seed)
        space_coords = np.random.rand(num_nodes, 2)
        dist = cdist(space_coords, space_coords)

        # create the spatial covariance matrix
        length_scale = 0.1
        nu = 1.5
        sigma = 1.0
        var_spatial = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                var_spatial[i, j] = self.matern_covariance(space_coords[i, :], space_coords[j, :], length_scale, nu, sigma)

        L_spatial = cholesky(var_spatial + 1e-6 * np.eye(num_nodes), lower=True)

        eta = rng.normal(0, 1, num_nodes)

        alpha = np.einsum('ij, j->i', L_spatial, eta)

        y = np.zeros((seq_len, num_nodes))

        epsilon = rng.normal(0, 1, size=(seq_len, num_nodes)) * 0.1

        for i in range(seq_len):
            y[i, :] = np.sin(i/12 + alpha) + epsilon[i, :]

        self.original_data['y'] = y


        time_coords = np.arange(0, seq_len)

        plt.figure()
        plt.plot(time_coords, y)
        plt.show()
        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index)

        space_coords, time_coords = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
        st_coords = np.stack([space_coords, time_coords], axis=-1)

        return df, dist, st_coords



    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)

    def matern_covariance(self, x1, x2, length_scale=1.0, nu=1.5, sigma=1.0):
        dist = np.linalg.norm(x1 - x2)
        if dist == 0:
            return sigma ** 2
        coeff1 = (2 ** (1 - nu)) / gamma(nu)
        coeff2 = (np.sqrt(2 * nu) * dist) / length_scale
        return sigma ** 2 * coeff1 * (coeff2 ** nu) * kv(nu, coeff2)



if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 5, 100
    dataset = Sine(num_nodes, seq_len)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




