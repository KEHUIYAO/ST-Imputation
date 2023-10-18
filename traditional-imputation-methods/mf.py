import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from fancyimpute import MatrixFactorization
from data.soil_moisture_sparse import SoilMoistureSparse



seed = 42
rng = np.random.RandomState(seed)

df = SoilMoistureSparse(mode='train')
df = df.original_data

y = df['y']
mask = df['mask']
X = df['X']
smap36 = df['original_X'][:, :, 0]

# randomly mask out rows with probability p = 0.2
p = 0.2
time_points_to_eval = rng.choice(y.shape[0], int(p*y.shape[0]), replace=False)
eval_mask = np.zeros_like(y)
eval_mask[time_points_to_eval, :] = 1

observed_mask = mask
eval_mask = eval_mask * observed_mask
training_mask = observed_mask - eval_mask

y_train = y.copy()
y_train[training_mask == 0] = np.nan
y_val = y.copy()
y_val[eval_mask == 0] = np.nan


y_imputed = MatrixFactorization().fit_transform(y_train)


mae = np.mean(np.abs(y_val - y_imputed)[eval_mask == 1])

print(f'MAE: {mae:.5f}')

mre = np.mean(np.abs(y_imputed - y_val)[eval_mask == 1] / np.abs(y_val[eval_mask == 1]))
print(f'MRE: {mre:.5f}')