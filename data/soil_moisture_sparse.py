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




class SoilMoistureSparse(PandasDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self, mode='train', seed=42):
        self.rng = np.random.RandomState(seed)
        self.original_data = {}
        df, dist, mask, st_coords_new, X_new, eval_mask_new = self.load(mode=mode)

        super().__init__(dataframe=df,
                         similarity_score="distance",
                         mask=mask,
                         attributes=dict(dist=dist,
                          st_coords=st_coords_new, covariates=X_new))

        # super().__init__(dataframe=df,
        #                  similarity_score="distance",
        #                  mask=mask,
        #                  attributes=dict(dist=dist,
        #                   st_coords=st_coords_new))

        self.set_eval_mask(eval_mask_new)





    def load(self, mode):

        if mode == 'train':
            date_start = '2016-01-01'
            date_end = '2020-12-31'
        else:
            date_start = '2021-01-01'
            date_end = '2022-12-31'

        df = pd.read_csv(os.path.join(current_dir, 'smap_1km.csv'))
        y = df.iloc[:, 4:]

        # transpose the dataframe
        y = y.T

        tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))

        y.index = pd.to_datetime(y.index)
        y = tmp.merge(y, left_index=True, right_index=True, how='left')

        y = y.values

        mask = ~np.isnan(y)
        mask = mask.astype(int)
        self.original_data['mask'] = mask

        rows, cols = y.shape

        p_missing = 0.2
        time_points_to_eval = self.rng.choice(rows, int(p_missing * rows), replace=False)
        eval_mask = np.zeros_like(y)
        eval_mask[time_points_to_eval, :] = 1

        self.original_data['eval_mask'] = eval_mask

        y_imputed = y.copy()
        # y_imputed[eval_mask == 1] = np.nan
        #
        # # impute using interpolation method
        # for i in range(cols):
        #     y_imputed[:, i] = pd.Series(y_imputed[:, i]).interpolate(method='linear', limit_direction='both').values

        y_imputed[np.isnan(y_imputed)] = 0

        # y_imputed[(eval_mask==1) & (mask==1)] = y[(eval_mask==1) & (mask==1)]

        y = y_imputed.copy()

        self.original_data['y'] = y

        y = pd.DataFrame(y)

        y.index = pd.to_datetime(y.index)



        # spatiotemporal coords
        space_coords, time_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        st_coords = np.stack([space_coords, time_coords], axis=-1)



        # spatiotemporal covariates
        covariates = ['smap_36km', 'prcp_1km', 'srad_1km', 'tmax_1km', 'tmin_1km', 'vp_1km']
        # covariates = ['prcp_1km', 'srad_1km', 'tmax_1km', 'tmin_1km', 'vp_1km']
        # covariates = ['smap_36km']

        # covariates = ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'SMAP_36km', 'elevation', 'slope', 'aspect', 'hillshade', 'clay', 'sand', 'bd', 'soc', 'LC']

        X = []
        for cov in covariates:
            x = pd.read_csv(os.path.join(current_dir, f'{cov}.csv'))
            x = x.iloc[:, 4:]
            x = x.T

            x.index = pd.to_datetime(x.index)
            tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))

            x = tmp.merge(x, left_index=True, right_index=True, how='left')

            x = x.values
            x_mask = ~np.isnan(x)
            x_mask = x_mask.astype(int)
            x[x_mask==0] = np.nanmean(x)

            X.append(x)
            X.append(x_mask)

        X = np.stack(X, axis=-1)

        # static features
        static_features = ['elevation', 'slope', 'aspect', 'hillshade', 'clay', 'sand', 'bd', 'soc', 'LC']
        tmp = pd.read_csv(os.path.join(current_dir, 'constant_grid.csv'))
        tmp = tmp.iloc[:, 4:].values  # (K, C)
        tmp = np.tile(tmp[np.newaxis, :, :], (X.shape[0], 1, 1))
        X = np.concatenate([X, tmp], axis=-1)





        self.original_data['original_X'] = X.copy()

        L, K, C = X.shape
        X = X.reshape((L * K, C))
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = X.reshape((L, K, C))

        self.original_data['X'] = X

        # if mode == 'train':
        #     y = y.iloc[:-365, :]
        #     X = X[:-365, :, :]
        #
        # elif mode == 'test':
        #     y = y.iloc[-365:, :]
        #     X = X[-365:, :, :]
        #
        # plt.figure()
        # plt.plot(np.arange(0, y.shape[0]), y.values, marker='o', linestyle='-')
        # plt.show()
        #


        unique_points = df.iloc[:, [0, 2, 3]]
        unique_points.columns = ['POINTID', 'x', 'y']

        sorted_x = np.sort(unique_points['x'].unique())
        sorted_y = np.sort(unique_points['y'].unique())

        pointid_dict = {}
        for i, row in unique_points.iterrows():
            pointid_dict[(row['x'], row['y'])] = row['POINTID']

        i = 0
        j = 0
        complete_split = []
        for j in range(0, 24, 12):
            for i in range(0, 24, 12):
                cur_split = []
                for jj in range(j, j+12):
                    for ii in range(i, i+12):
                        cur_split.append(pointid_dict[(sorted_x[ii], sorted_y[jj])])
                complete_split.append(cur_split)

        complete_split = np.array(complete_split, dtype=int)

        df_new = []
        st_coords_new = []
        eval_mask_new = []
        mask_new = []
        X_new = []
        for i in range(complete_split.shape[0]):
            # select all columns in the split
            cur_split = complete_split[i]
            ind = [x-1 for x in cur_split]
            cur_df = y.iloc[:, ind]
            df_new.append(cur_df)

            st_coords_new.append(st_coords[:, ind, :])
            X_new.append(X[:, ind, :])
            eval_mask_new.append(eval_mask[:, ind])
            mask_new.append(mask[:, ind])


        df_array = [cur_df.values for cur_df in df_new]
        df_array = np.concatenate(df_array, axis=0)

        df_new = pd.DataFrame(df_array)
        df_new.index = pd.to_datetime(df_new.index)

        st_coords_new = np.concatenate(st_coords_new, axis=0)
        X_new = np.concatenate(X_new, axis=0)
        eval_mask_new = np.concatenate(eval_mask_new, axis=0)
        mask_new = np.concatenate(mask_new, axis=0)



        dist = []
        for j in range(6):
            for i in range(6):
                dist.append([sorted_x[i], sorted_y[j]])
        dist = np.array(dist)
        dist = cdist(dist, dist)


        return df_new, dist, mask_new, st_coords_new, X_new, eval_mask_new

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
    dataset = SoilMoistureSparse()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
