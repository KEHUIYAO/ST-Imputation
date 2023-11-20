import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler

from data.soil_moisture_sparse import SoilMoistureSparse



df = SoilMoistureSparse(mode='train', seed = 42)
df = df.original_data
y = df['y']
X = df['X']

# save X and y (numpy array) to csv file
np.savetxt('data/X.csv', X, delimiter=',')
np.savetxt('data/y.csv', y, delimiter=',')


