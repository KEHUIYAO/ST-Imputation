import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from .utils import positional_encoding

class SpatiotemporalBasisFunctions:
    def __init__(self):
        pass

    @staticmethod
    def bisquare1d_basis(s, t, center, threshold):
        return np.where(
            np.abs(t-center) < threshold,
            (1 - (np.abs(t-center) / threshold) ** 2) ** 2,
            0
        )

    @staticmethod
    def bisquare2d_basis(s, t, center, threshold):
        return np.where(
            np.linalg.norm(s-center, axis=1) < threshold,
            (1 - (np.linalg.norm(s-center, axis=1) / threshold) ** 2) ** 2,
            0
        )

    @staticmethod
    def cosine_basis(s, t, n):
        return np.cos(n*t)

    @staticmethod
    def generate_basis_function(name, **kwargs):
        if name == "bisquare1d":
            def wrapper(s, t):
                return SpatiotemporalBasisFunctions.bisquare1d_basis(s, t, **kwargs)
        elif name == "bisquare2d":
            def wrapper(s, t):
                return SpatiotemporalBasisFunctions.bisquare2d_basis(s, t, **kwargs)
        elif name == "cosine":
            def wrapper(s, t):
                return SpatiotemporalBasisFunctions.cosine_basis(s, t, **kwargs)
        else:
            raise ValueError("Unknown basis function: {}".format(name))

        return wrapper




class DescriptiveST(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self,
                 num_nodes,
                 seq_len,
                 covariate_size=10,
                 linear_effect=True,
                 num_time_basis_functions=100,
                 num_space_basis_functions=100,
                 seed=42):
        df, dist, covariates = self.load(num_nodes, seq_len, covariate_size, linear_effect, num_time_basis_functions, num_space_basis_functions, seed)
        temporal_encoding= positional_encoding(seq_len, 1, 4).squeeze(1)

        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, covariates=covariates, temporal_encoding=temporal_encoding))


    def load(self, num_nodes, seq_len, covariate_size, linear_effect, num_time_basis_functions, num_space_basis_functions, seed):
        rng = np.random.RandomState(seed)
        time_coords = np.arange(0, seq_len)
        space_coords = rng.rand(num_nodes, 2)
        self.time_coords = time_coords
        self.space_coords = space_coords
        dist = cdist(space_coords, space_coords)

        external_covariates, fixed_effect = self.generate_spatiotemporal_fixed_effect(num_nodes, seq_len, covariate_size, linear_effect, seed)
        spatiotemporal_covariates, random_effect = self.generate_random_effect_with_basis_functions(num_nodes, seq_len, num_time_basis_functions, num_space_basis_functions, seed)
        epsilon = rng.normal(0, 1, size=num_nodes*seq_len)
        epsilon = epsilon * 0.2

        y = fixed_effect + random_effect + epsilon
        # y = random_effect + epsilon
        # y = epsilon
        y = y.reshape(num_nodes, seq_len)
        external_covariates = external_covariates.reshape(num_nodes, seq_len, covariate_size)
        #spatiotemporal_covariates = spatiotemporal_covariates.reshape(num_nodes, seq_len, -1)
        #covariates = np.concatenate([external_covariates, spatiotemporal_covariates], axis=2)
        covariates = external_covariates
        covariates = covariates.transpose(1, 0, 2)


        y = y.T

        plt.figure()
        plt.plot(time_coords, y)
        plt.show()

        df = pd.DataFrame(y)
        df.index = pd.to_datetime(df.index)

        return df, dist, covariates


    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)


    def generate_spatiotemporal_fixed_effect(self, num_nodes, seq_len, covariate_size, linear_effect, seed):

        rng = np.random.RandomState(seed=seed)

        n_samples = num_nodes * seq_len

        if linear_effect:
            n_informative = round(covariate_size/2)
            X, y = make_regression(n_samples=n_samples, n_features=covariate_size, n_informative=n_informative, random_state=rng)
            scaler = StandardScaler()
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            y = y * 0.5

        else:
            X = rng.rand(n_samples, covariate_size)
            hidden_dim = 64
            W = [rng.uniform(-np.sqrt(1 / covariate_size), np.sqrt(1 / covariate_size), size=[covariate_size, hidden_dim]), rng.uniform(-np.sqrt(1 / hidden_dim), np.sqrt(1 / hidden_dim), size=[hidden_dim, 1])]

            def relu(x):
                return np.maximum(x, 0)

            y = relu(np.einsum('nj, ji->ni', X, W[0]))
            y = np.einsum('ni, ij->nj', y, W[1])
            y = y.flatten()

        return X, y


    def generate_random_effect_with_basis_functions(self, num_nodes, seq_len, num_time_basis_functions, num_space_basis_functions, seed):
        rng = np.random.RandomState(seed=seed)
        time_coords = self.time_coords
        space_coords = self.space_coords


        time_basis_functions = [SpatiotemporalBasisFunctions.generate_basis_function("bisquare1d", center=center, threshold=threshold) for (center, threshold) in zip(rng.uniform(0, seq_len, size=num_time_basis_functions), rng.uniform(0, seq_len, size=num_time_basis_functions))]

        space_basis_functions = [SpatiotemporalBasisFunctions.generate_basis_function("bisquare2d", center=center, threshold=threshold) for (center, threshold) in zip(rng.uniform(0, 1, size=(num_space_basis_functions, 2)), rng.uniform(0, np.sqrt(2), size=num_space_basis_functions))]

        # sample coefficients from normal distribution
        coefficients = rng.normal(0, 1, size=num_time_basis_functions + num_space_basis_functions + num_time_basis_functions * num_space_basis_functions)

        time_coords = np.tile(time_coords, num_nodes)
        space_coords = np.repeat(space_coords, seq_len, axis=0)

        X_space = np.stack([func(space_coords, time_coords) for func in space_basis_functions], axis=1)
        X_time = np.stack([func(space_coords, time_coords) for func in time_basis_functions], axis=1)
        X_space_time = np.zeros([num_nodes*seq_len, num_time_basis_functions * num_space_basis_functions])
        for i in range(num_space_basis_functions):
            for j in range(num_time_basis_functions):
                X_space_time[:, num_time_basis_functions*i+j] = X_space[:, i] * X_time[:, j]

        X = np.concatenate([X_space, X_time, X_space_time], axis=1)

        y = np.einsum('ij, j->i', X, coefficients)

        return X, y


if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values

    num_nodes, seq_len = 36, 200
    dataset = DescriptiveST(num_nodes, seq_len, seed=42)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




