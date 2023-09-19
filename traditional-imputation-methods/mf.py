import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from fancyimpute import MatrixFactorization



current_dir = os.path.dirname(os.path.abspath(__file__))
seed = 42
rng = np.random.RandomState(seed)
data_path = os.path.join(current_dir, '../data/SMAP_Climate_In_Situ_TxSON.csv')


df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# training data
y = df[df['Date'].dt.year == 2016].copy()
y = y.pivot(index='Date', columns='POINTID', values='SMAP_1km').values

covariates = ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'SMAP_36km']
X = []
for cov in covariates:
    x = df[df['Date'].dt.year == 2017].pivot(index='Date', columns='POINTID', values=cov).values
    # impute missing values with mean
    x[np.isnan(x)] = np.nanmean(x)
    X.append(x)

X = np.stack(X, axis=-1)
L, K, C = X.shape
X = X.reshape((L*K, C))
X = StandardScaler().fit_transform(X)
X = X.reshape((L, K, C))

# randomly mask out rows with probability p = 0.2
p = 0.2
time_points_to_eval = rng.choice(y.shape[0], int(p*y.shape[0]), replace=False)
eval_mask = np.zeros_like(y)
eval_mask[time_points_to_eval, :] = 1

observed_mask = np.ones_like(y)
observed_mask[np.isnan(y)] = 0
eval_mask = eval_mask * observed_mask
training_mask = observed_mask - eval_mask

y_train = y.copy()
y_train[training_mask == 0] = np.nan
y_val = y.copy()
y_val[eval_mask == 0] = np.nan

y_imputed = MatrixFactorization().fit_transform(y_train)


mae = np.mean(np.abs(y_val - y_imputed)[eval_mask == 1])

print(f'MAE: {mae:.5f}')