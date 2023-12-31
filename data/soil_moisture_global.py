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




class SoilMoistureHB(PandasDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self, mode='train', seed=42):
        self.rng = np.random.RandomState(seed)
        self.original_data = {}
        df, dist, mask, st_coords_new, X_new, eval_mask_new = self.load(mode=mode)

        # super().__init__(dataframe=df,
        #                  similarity_score="distance",
        #                  mask=mask,
        #                  attributes=dict(dist=dist,
        #                   st_coords=st_coords_new, covariates=X_new))

        super().__init__(dataframe=df,
                         similarity_score="distance",
                         mask=mask,
                         attributes=dict(dist=dist,
                          st_coords=st_coords_new))

        self.set_eval_mask(eval_mask_new)





    def load(self, mode):

        if mode == 'train':
            date_start = '2016-01-01'
            date_end = '2018-12-31'
        else:
            date_start = '2019-01-01'
            date_end = '2019-12-31'

        filename_list = ['smap_hb_1km_sample_1.csv', 'smap_hb_1km_sample_2.csv', 'smap_hb_1km_sample_3.csv', 'smap_hb_1km_sample_4.csv', 'smap_hb_1km_sample_5.csv']
        y_list = []
        eval_mask_list = []
        mask_list = []
        for filename in filename_list:
            df = pd.read_csv(os.path.join(current_dir, filename))
            y = df.iloc[:, 3:]
            # transpose the dataframe
            y = y.T
            tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
            y.index = pd.to_datetime(y.index)
            y = tmp.merge(y, left_index=True, right_index=True, how='left')
            y = y.values
            mask = ~np.isnan(y)
            mask = mask.astype(int)
            rows, cols = y.shape

            p_missing = 0.8
            ################# missing completely for selected time point ##################
            time_points_to_eval = self.rng.choice(rows, int(p_missing * rows), replace=False)
            eval_mask = np.zeros_like(y)
            eval_mask[time_points_to_eval, :] = 1

            ################# missing completely for selected time point ##################

            # ################## missing at random ##################
            # eval_mask = np.zeros_like(y)
            # # randomly mask p_missing of the data
            # eval_mask[self.rng.rand(*y.shape) < p_missing] = 1
            # self.original_data['eval_mask'] = eval_mask
            # ################## missing at random ##################
            y_imputed = y.copy()
            y_imputed[np.isnan(y_imputed)] = 0
            y_list.append(y_imputed.copy())
            eval_mask_list.append(eval_mask.copy())
            mask_list.append(mask.copy())










        y = np.concatenate(y_list, axis=0)
        self.original_data['y'] = y
        rows, cols = y.shape
        eval_mask = np.concatenate(eval_mask_list, axis=0)
        mask = np.concatenate(mask_list, axis=0)
        y = pd.DataFrame(y)
        y.index = pd.to_datetime(y.index)




        # spatiotemporal coords
        space_coords, time_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        st_coords = np.stack([space_coords, time_coords], axis=-1)



        unique_points = df.iloc[:, [0, 1, 2]]
        unique_points.columns = ['POINTID', 'x', 'y']

        sorted_x = np.sort(unique_points['x'].unique())
        sorted_y = np.sort(unique_points['y'].unique())

        pointid_dict = {}
        for i, row in unique_points.iterrows():
            pointid_dict[(row['x'], row['y'])] = row['POINTID']

        i = 0
        j = 0
        complete_split = []
        for j in range(0, 36, 12):
            for i in range(0, 36, 12):
                cur_split = []
                for jj in range(j, j+12):
                    for ii in range(i, i+12):
                        cur_split.append(pointid_dict[(sorted_x[ii], sorted_y[jj])])
                complete_split.append(cur_split)

        # i = 0
        # j = 0
        # complete_split = []
        # for j in range(0, 36, 6):
        #     for i in range(0, 36, 6):
        #         cur_split = []
        #         for jj in range(j, j + 6):
        #             for ii in range(i, i + 6):
        #                 cur_split.append(pointid_dict[(sorted_x[ii], sorted_y[jj])])
        #         complete_split.append(cur_split)




        complete_split = np.array(complete_split, dtype=np.int32)

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
            eval_mask_new.append(eval_mask[:, ind])
            mask_new.append(mask[:, ind])


        df_array = [cur_df.values for cur_df in df_new]
        df_array = np.concatenate(df_array, axis=0)

        df_new = pd.DataFrame(df_array)
        df_new.index = pd.to_datetime(df_new.index)

        st_coords_new = np.concatenate(st_coords_new, axis=0)
        eval_mask_new = np.concatenate(eval_mask_new, axis=0)
        mask_new = np.concatenate(mask_new, axis=0)



        dist = []
        for j in range(12):
            for i in range(12):
                dist.append([sorted_x[i], sorted_y[j]])
        dist = np.array(dist)
        dist = cdist(dist, dist)

        X_new = None


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
    dataset = SoilMoistureHB()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
