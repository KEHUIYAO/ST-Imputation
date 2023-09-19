import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


current_dir = os.path.dirname(os.path.abspath(__file__))
seed = 42
rng = np.random.RandomState(seed)
data_path = os.path.join(current_dir, 'data/SMAP_Climate_In_Situ_TxSON.csv')


df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# training data
y = df[df['Date'].dt.year == 2016].copy()
y = y.pivot(index='Date', columns='POINTID', values='SMAP_1km').values

covariates = ['prcp', 'srad', 'tmax', 'tmin', 'vp', 'SMAP_36km']
X = []
for cov in covariates:
    x = df[df['Date'].dt.year == 2016].pivot(index='Date', columns='POINTID', values=cov).values
    # impute missing values with mean
    x[np.isnan(x)] = np.nanmean(x)
    X.append(x)

X = np.stack(X, axis=-1)

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

X_train = X.copy()
X_train[training_mask == 0] = np.nan
X_val = X.copy()
X_val[eval_mask == 0] = np.nan


y_train = y_train[training_mask == 1]
X_train = X_train[training_mask == 1]
y_val = y_val[eval_mask == 1]
X_val = X_val[eval_mask == 1]

# Train a random forest regressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)


y_val_pred = regressor.predict(X_val)

# mae
mae = np.nanmean(np.abs(y_val_pred - y_val))
print(f'MAE: {mae:.5f}')









