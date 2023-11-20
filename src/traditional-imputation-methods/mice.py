import os
import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data.soil_moisture_sparse import SoilMoistureSparse
from data.soil_moisture_hb import SoilMoistureHB


n_rep = 5
seed_list = [i for i in range(n_rep)]

result = {'validation': {'mae': [], 'mre': []}, 'test': {'mae': [], 'mre': []}}


for seed in seed_list:
    print('run the {}th simulation.'.format(seed+1))

    # df = SoilMoistureSparse(mode='train', seed = seed)
    df = SoilMoistureHB(mode='train', seed=seed)
    df = df.original_data
    y = df['y']
    observed_mask = df['mask']
    X = df['X']
    eval_mask = df['eval_mask']

    eval_mask = eval_mask * observed_mask
    training_mask = observed_mask - eval_mask

    y_train = y.copy()
    y_train[training_mask == 0] = np.nan
    y_val = y.copy()
    y_val[eval_mask == 0] = np.nan

    y_val_pred = y_train.copy()
    mice_imputer = IterativeImputer()
    tmp = np.concatenate([y_train.reshape(-1, 1), X.reshape(-1, X.shape[-1])], axis=-1)

    # sample 1000 rows for training
    sample_idx = np.random.choice(tmp.shape[0], 100000, replace=False)
    tmp_subset = tmp[sample_idx, :]

    mice_imputer.fit(tmp_subset)
    y_val_pred = mice_imputer.transform(tmp)[:, 0]

    y_val_pred = y_val_pred.reshape(y_train.shape)

    mae = np.mean(np.abs(y_val_pred - y_val)[eval_mask == 1])

    print(f'MAE: {mae:.5f}')

    mre = np.mean(np.abs(y_val_pred - y_val)[eval_mask == 1] / np.abs(y_val[eval_mask == 1]))
    print(f'MRE: {mre:.5f}')

    # save result
    result['validation']['mae'].append(mae)
    result['validation']['mre'].append(mre)


    df = SoilMoistureSparse(mode='test', seed=seed)
    df = df.original_data

    y = df['y']
    observed_mask = df['mask']
    X = df['X']
    eval_mask = df['eval_mask']


    eval_mask = eval_mask * observed_mask
    training_mask = observed_mask - eval_mask

    y_train = y.copy()
    y_train[training_mask == 0] = np.nan
    y_val = y.copy()
    y_val[eval_mask == 0] = np.nan

    y_val_pred = y_train.copy()
    mice_imputer = IterativeImputer()
    tmp = np.concatenate([y_train.reshape(-1, 1), X.reshape(-1, X.shape[-1])], axis=-1)

    # sample 1000 rows for training
    sample_idx = np.random.choice(tmp.shape[0], 100000, replace=False)
    tmp_subset = tmp[sample_idx, :]

    mice_imputer.fit(tmp_subset)
    y_val_pred = mice_imputer.transform(tmp)[:, 0]

    y_val_pred = y_val_pred.reshape(y_train.shape)

    mae = np.mean(np.abs(y_val_pred - y_val)[eval_mask == 1])

    print(f'MAE: {mae:.5f}')

    mre = np.mean(np.abs(y_val_pred - y_val)[eval_mask == 1] / np.abs(y_val[eval_mask == 1]))
    print(f'MRE: {mre:.5f}')

    # save result
    result['test']['mae'].append(mae)
    result['test']['mre'].append(mre)




# compute mean and std
for key in result.keys():
    for sub_key in result[key].keys():
        result[key][sub_key] = np.array(result[key][sub_key])
        print(f'{key} {sub_key}: {result[key][sub_key].mean():.5f} +/- {result[key][sub_key].std():.5f}')

