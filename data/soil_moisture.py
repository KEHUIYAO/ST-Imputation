import pandas as pd

from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np

from .utils import positional_encoding

from scipy.spatial.distance import cdist
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Insitu_gap_filling_data.csv')


class SoilMoisture(PandasDataset):
    similarity_options = {'distance'}

    def __init__(self):
        df, dist, mask, temporal_encoding = self.load()


        super().__init__(dataframe=df,
                         similarity_score="distance",
                         mask=mask,
                         attributes=dict(dist=dist, temporal_encoding=temporal_encoding))

    def load(self):
        df = pd.read_csv(data_path)

        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        y = df.pivot(index='Date', columns='POINTID', values='SMAP_1km')



        time_coords = np.arange(0, y.shape[0])
        plt.figure()
        plt.plot(time_coords, y.values)
        plt.show()

        temporal_encoding = positional_encoding(730, 1, 4).squeeze(1)
        temporal_encoding = np.tile(temporal_encoding, (36, 1))

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
        for i in range(complete_split.shape[0]):
            # select all columns in the split
            cur_split = complete_split[i]
            cur_df = y[cur_split]
            df_new.append(cur_df)

        df_array = [cur_df.values for cur_df in df_new]
        df_array = np.concatenate(df_array, axis=0)

        df_new = pd.DataFrame(df_array)
        df_new.index = pd.to_datetime(df_new.index)

        mask = df_new.notnull().astype(int).values

        df_new = df_new.fillna(0)

        dist = []
        for j in range(6):
            for i in range(6):
                dist.append([sorted_x[i], sorted_y[j]])
        dist = np.array(dist)
        dist = cdist(dist, dist)

        return df_new, dist, mask, temporal_encoding

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)





if __name__ == '__main__':
    from tsl.ops.imputation import add_missing_values
    dataset = SoilMoisture()
    add_missing_values(dataset, p_fault=0, p_noise=0.25, min_seq=12,
                       max_seq=12 * 4, seed=56789)

    print(dataset.training_mask.shape)
    print(dataset.eval_mask.shape)

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False)

    print(adj)
