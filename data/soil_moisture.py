import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


class SoilMoisture(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self,
                 num_nodes,
                 seq_len,
                 seed=42):
        df, dist, covariates = self.load(num_nodes, seq_len, seed)

        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, covariates=covariates))


    def load(self, num_nodes, seq_len, seed):
        rng = np.random.RandomState(seed)
        time_coords = np.arange(0, seq_len)
        space_coords = rng.rand(num_nodes, 2)
        self.time_coords = time_coords
        self.space_coords = space_coords
        dist = cdist(space_coords, space_coords)

        y = np.zeros((seq_len, num_nodes))

        prcp = rng.uniform(0, 1, size=(seq_len, num_nodes))
        prcp_obs = prcp + rng.normal(0, 0.1, size=(seq_len, num_nodes))
        mask = rng.uniform(0, 1, size=(seq_len, num_nodes))
        prcp[mask > 0.1] = 0
        prcp_obs[mask > 0.1] = 0


        vp = rng.uniform(0, 0.1, size=(seq_len, num_nodes))
        vp_obs = vp + rng.normal(0, 0.01, size=(seq_len, num_nodes))


        y[0, :] = rng.uniform(0, 1, size=(num_nodes))
        for t in range(1, seq_len):
            y[t, :] = y[t-1, :] * 0.8 + prcp_obs[t-1, :] - vp_obs[t-1, :]

        y = y + rng.normal(0, 0.1, size=(seq_len, num_nodes))


        plt.figure()
        plt.plot(time_coords, y)
        plt.show()

        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index)

        covariates = np.stack([prcp_obs, vp_obs], axis=2)

        return df, dist, covariates


    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)




if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 1, 20000
    dataset = SoilMoisture(num_nodes, seq_len, seed=42)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




