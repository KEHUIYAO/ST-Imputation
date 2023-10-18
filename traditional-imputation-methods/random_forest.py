import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from data.soil_moisture_sparse import SoilMoistureSparse



seed = 42
rng = np.random.RandomState(seed)

df = SoilMoistureSparse(mode='train')
df = df.original_data

y = df['y']
mask = df['mask']
X = df['X']


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


y_train = y_train[training_mask == 1]
X_train = X[training_mask == 1].copy()
y_val = y_val[eval_mask == 1]
X_val = X[eval_mask == 1].copy()

# Train a random forest regressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)


print(regressor.feature_importances_)

y_val_pred = regressor.predict(X_val)




# mae
mae = np.nanmean(np.abs(y_val_pred - y_val))
print(f'MAE: {mae:.5f}')

mre = np.mean(np.abs(y_val_pred - y_val)/ np.abs(y_val))
print(f'MRE: {mre:.5f}')









