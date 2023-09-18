import os
import pandas as pd

import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_dir, 'data/SMAP_Climate_In_Situ_TxSON.csv')
df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

df = df[df['Date'].dt.year == 2016]


y = df.pivot(index='Date', columns='POINTID', values='SMAP_1km')

y = y.values

# minimum value of y excluding nan
min_y = np.nanmin(y)

# maximum value of y excluding nan
max_y = np.nanmax(y)

# randomly mask out rows with probability p = 0.2
p = 0.2
rng = np.random.RandomState(42)
time_points_to_eval = rng.choice(y.shape[0], int(p*y.shape[0]), replace=False)
eval_mask = np.zeros_like(y)
eval_mask[time_points_to_eval, :] = 1

y_train = y.copy()
y_train[eval_mask == 1] = np.nan

# proportion of missing values in y_train
p_missing = np.sum(np.isnan(y_train)) / np.prod(y_train.shape)
print(p_missing)

# min and max of y_train
min_y_train = np.nanmin(y_train)
max_y_train = np.nanmax(y_train)

# save y and y_train to csv
np.savetxt(os.path.join(current_dir, 'data/y_train.csv'), y_train, delimiter=',')
np.savetxt(os.path.join(current_dir, 'data/y.csv'), y, delimiter=',')

