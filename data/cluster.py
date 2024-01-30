import pandas as pd
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin

from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky





class Cluster(PandasDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self, num_nodes, seq_len, seed=42):
        self.original_data = {}
        df, dist, st_coords = self.load(num_nodes, seq_len, seed)
        super().__init__(dataframe=df, similarity_score="distance", attributes=dict(dist=dist, st_coords=st_coords))
        eval_mask = np.zeros((seq_len, num_nodes))
        p_missing = 0.9
        rng = np.random.RandomState(seed)
        time_points_to_eval = rng.choice(seq_len, int(p_missing * seq_len), replace=False)
        eval_mask[time_points_to_eval, :] = 1
        self.original_data['eval_mask'] = eval_mask
        self.set_eval_mask(eval_mask)





    def load(self, num_nodes, seq_len, seed):

        rng = np.random.RandomState(seed)
        space_coords = np.random.rand(num_nodes, 2)
        dist = cdist(space_coords, space_coords)
        y = np.zeros((seq_len, num_nodes))
        rng = np.random.RandomState(seed)

        for i in range(num_nodes):
              # For reproducibility
            noise_level = 0.2  # Noise level

            # Randomly determine the number of segments and their lengths
            num_segments = np.random.randint(50, 100) # Random number of segments between 10 and 20
            segment_lengths = np.random.choice(range(20, 50), num_segments)  # Random segment lengths between 20 and 50
            segment_lengths = np.round(segment_lengths / sum(segment_lengths) * seq_len).astype(
                int)  # Adjust to match seq_len

            # Adjust the last segment to ensure the total length matches seq_len
            segment_lengths[-1] = seq_len - sum(segment_lengths[:-1])

            segments = []
            last_value = 0
            for length in segment_lengths:
                pattern_type = np.random.choice(['upward', 'downward', 'stable'])

                if pattern_type == 'upward':
                    trend = np.linspace(last_value, last_value + 4, length)
                elif pattern_type == 'downward':
                    trend = np.linspace(last_value, last_value - 4, length)
                else:  # stable
                    trend = np.full(length, last_value)

                noise = rng.normal(0, noise_level, length)
                segment = trend + noise
                segments.append(segment)
                last_value = segment[-1]  # Update last value for the next segment

            # Combine segments
            time_series = np.concatenate(segments)
            y[:, i] = time_series



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

    num_nodes, seq_len = 5, 4000
    dataset = Cluster(num_nodes, seq_len)
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)




