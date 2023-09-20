import pandas as pd

from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin

from tsl.ops.similarities import gaussian_kernel

from tsl.data.datamodule.splitters import Splitter
import matplotlib.pyplot as plt
import numpy as np

from .utils import positional_encoding

from scipy.spatial.distance import cdist
import os
from sklearn.preprocessing import StandardScaler


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'SMAP_Climate_In_Situ_TxSON.csv')
#data_path = os.path.join(current_dir, 'SMAP_Climate_In_Situ_Kenaston.csv')


class SoilMoistureSparse(PandasDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self, mode='train', seed=42):
        self.seed = seed
        df, dist, mask, st_coords_new, temporal_encoding, eval_mask_new, X_new = self.load(mode=mode)
        super().__init__(dataframe=df,
                         similarity_score="distance",
                         mask=mask,
                         attributes=dict(dist=dist, temporal_encoding=temporal_encoding,
                          st_coords=st_coords_new, covariates=X_new))

        self.set_eval_mask(eval_mask_new)


    def load(self, mode):
        df = pd.read_csv(data_path)

        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')


        if mode == 'train':
            # select data from year 2016
            df = df[df['Date'].dt.year == 2016]
        elif mode == 'test':
            # select data from year 2017
            df = df[df['Date'].dt.year == 2017]

        y = df.pivot(index='Date', columns='POINTID', values='SMAP_1km')

        covariates = ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'SMAP_36km']
        X = []
        for cov in covariates:
            x = df.pivot(index='Date', columns='POINTID', values=cov).values
            # impute missing values with mean
            x[np.isnan(x)] = np.nanmean(x)
            X.append(x)

        X = np.stack(X, axis=-1)
        L, K, C = X.shape
        X = X.reshape((L * K, C))
        X = StandardScaler().fit_transform(X)
        X = X.reshape((L, K, C))






        temporal_encoding = positional_encoding(365, 1, 4).squeeze(1)
        temporal_encoding = np.tile(temporal_encoding, (36, 1))

        rows, cols = y.shape
        space_coords, time_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        st_coords = np.stack([space_coords, time_coords], axis=-1)

        rng = np.random.RandomState(self.seed)

        p_missing = 0.2
        time_points_to_eval = rng.choice(rows, int(p_missing * rows), replace=False)
        eval_mask = np.zeros((rows, cols))
        eval_mask[time_points_to_eval, :] = 1


        plt.figure()
        plt.plot(np.arange(0, y.shape[0]), y.values, marker='o', linestyle='-')
        plt.show()

        unique_points = df.drop_duplicates(subset='POINTID')[['x', 'y', 'POINTID']]

        sorted_x = np.sort(unique_points['x'].unique())
        sorted_y = np.sort(unique_points['y'].unique())

        pointid_dict = {}
        for i, row in unique_points.iterrows():
            pointid_dict[(row['x'], row['y'])] = row['POINTID']

        i = 0
        j = 0
        complete_split = []
        for j in range(0, 36, 6):
            for i in range(0, 36, 6):
                cur_split = []
                for jj in range(j, j+6):
                    for ii in range(i, i+6):
                        cur_split.append(pointid_dict[(sorted_x[ii], sorted_y[jj])])
                complete_split.append(cur_split)

        complete_split = np.array(complete_split)

        df_new = []
        st_coords_new = []
        eval_mask_new = []
        X_new = []
        for i in range(complete_split.shape[0]):
            # select all columns in the split
            cur_split = complete_split[i]
            cur_df = y[cur_split]
            df_new.append(cur_df)

            ind = [list(y.columns).index(col) for col in cur_split]
            st_coords_new.append(st_coords[:, ind, :])
            eval_mask_new.append(eval_mask[:, ind])
            X_new.append(X[:, ind, :])


        df_array = [cur_df.values for cur_df in df_new]
        df_array = np.concatenate(df_array, axis=0)

        df_new = pd.DataFrame(df_array)
        df_new.index = pd.to_datetime(df_new.index)

        st_coords_new = np.concatenate(st_coords_new, axis=0)
        eval_mask_new = np.concatenate(eval_mask_new, axis=0)
        X_new = np.concatenate(X_new, axis=0)

        mask = df_new.notnull().astype(int).values
        df_new = df_new.fillna(0)

        dist = []
        for j in range(6):
            for i in range(6):
                dist.append([sorted_x[i], sorted_y[j]])
        dist = np.array(dist)
        dist = cdist(dist, dist)

        return df_new, dist, mask, st_coords_new, temporal_encoding, eval_mask_new, X_new

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)


    def get_splitter(self, method=None, **kwargs):
        return SoilMoistureSplitter(kwargs.get('val_len'), kwargs.get('test_len')   )


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
        parser.add_argument('--val-len', type=float or int, default=0.1)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values
    dataset = SoilMoistureSparse()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
