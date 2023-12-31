import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler



current_dir = os.path.dirname(os.path.abspath(__file__))
seed = 42
rng = np.random.RandomState(seed)
data_path = os.path.join(current_dir, '../../data/SMAP_Climate_In_Situ_TxSON.csv')


df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# training data
y = df[df['Date'].dt.year == 2017].copy()
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



y_imputed = y_train.copy()

for i in range(y_train.shape[1]):
    tmp = np.concatenate([y_train[:, i][:, np.newaxis], X[:, i, :]], axis=-1)
    tmp_imputed = KNN(k=5).fit_transform(tmp)
    y_imputed[:, i] = tmp_imputed[:, 0]




mae = np.mean(np.abs(y_val - y_imputed)[eval_mask == 1])

print(f'MAE: {mae:.5f}')


# in-situ data
data_path = os.path.join(current_dir, '../../data/Insitu_gap_filling_data.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')


y2 = df[df['Date'].dt.year == 2017].copy()
y2 = y2.pivot(index='Date', columns='POINTID', values='SMAP_1km').values
observed_mask2 = np.ones_like(y2)
observed_mask2[np.isnan(y2)] = 0


# calculate ubrmse
bias = np.mean((y2 - y)[observed_mask*observed_mask2 == 1])
tmp = (y2-y)**2
tmp = np.mean(tmp[observed_mask*observed_mask2==1])
ubrmse = np.sqrt(tmp - bias**2)

print(f'UBRMSE: {ubrmse:.5f}')

# calculate correlation
corr = np.corrcoef(y2[observed_mask*observed_mask2 == 1], y[observed_mask*observed_mask2 == 1])[0, 1]
print(f'Correlation: {corr:.5f}')


# calculate ubrmse
bias = np.mean((y2 - y_imputed)[observed_mask2 == 1])
tmp = (y2-y_imputed)**2
tmp = np.mean(tmp[observed_mask2==1])
ubrmse = np.sqrt(tmp - bias**2)

print(f'UBRMSE: {ubrmse:.5f}')

# calculate correlation
corr = np.corrcoef(y2[observed_mask2 == 1], y_imputed[observed_mask2 == 1])[0, 1]
print(f'Correlation: {corr:.5f}')

