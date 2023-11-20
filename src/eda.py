import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

####################### plot time-varying data #######################
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(7,7), sharex=True)

df = pd.read_csv('../data/smap_1km.csv')
y = df.iloc[:, 4:]

# transpose the dataframe
y = y.T
date_start='2016-01-01'
date_end='2020-12-31'
tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))

y.index = pd.to_datetime(y.index)
y = tmp.merge(y, left_index=True, right_index=True, how='left')

y = y.values

# spatiotemporal covariates
covariates = ['smap_36km', 'prcp_1km', 'srad_1km', 'tmax_1km', 'tmin_1km', 'vp_1km']



X = []
for cov in covariates:
    x = pd.read_csv(f'data/{cov}.csv')
    x = x.iloc[:, 4:]
    x = x.T

    x.index = pd.to_datetime(x.index)
    tmp = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))

    x = tmp.merge(x, left_index=True, right_index=True, how='left')

    x = x.values

    X.append(x)

if len(X) > 0:
    X = np.stack(X, axis=-1)

date_range = pd.date_range(date_start, date_end)

k = 0

axes[0].plot(date_range, y[:, k], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[0].set_ylabel('value')
axes[0].set_title('SMAP 1km')

axes[1].plot(date_range, X[:, k, 0], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[1].set_ylabel('value')
axes[1].set_title('SMAP 36km')


axes[2].plot(date_range, X[:, k, 1], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[2].set_ylabel('value')
axes[2].set_title('prcp')

axes[3].plot(date_range, X[:, k, 2], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[3].set_ylabel('value')
axes[3].set_title('srad')


axes[4].plot(date_range, X[:, k, 3], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[4].set_ylabel('value')
axes[4].set_title('tmax')

axes[5].plot(date_range, X[:, k, 4], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[5].set_ylabel('value')
axes[5].set_title('tmin')

axes[6].plot(date_range, X[:, k, 5], color = 'k',marker = 'o', linestyle='None', markersize=1)
axes[6].set_ylabel('value')
axes[6].set_title('vp')
plt.tight_layout()
plt.show()


fig.savefig('./writings/figure/eda_time_varying.png', dpi=300, bbox_inches='tight')


######################################################################


######################## plot static data #######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

static = pd.read_csv('../data/constant_grid.csv')

lc = static[['coords.x1', 'coords.x2', 'LC']]

fig, ax = plt.subplots(figsize=(7,7))
values = np.unique(lc['LC'].values)
print(values)

# Adjust the main plot to make space for the legend on the right
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Use discrete colors for the scatter plot
scatter = ax.scatter(lc['coords.x1'], lc['coords.x2'], c=lc['LC'], cmap='tab20c')

# Create the legend and place it to the right of the axes
legend_labels = ['Woody savannas', 'Savannas', 'Grasslands', 'Croplands', 'Urban and Built-up Lands']
legend = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="LC")

ax.set_title('Land Cover Classification')

plt.show()

fig.savefig('./writings/figure/eda_static.png', dpi=300, bbox_inches='tight')

######################################################################

########################## plot error across space ######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_dir = '../experiments/log/soil_moisture_sparse_point/st_transformer/20231103T174145_0/in_sample.npz'

output = np.load(log_dir)

y_hat, y_true, observed_mask, eval_mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['observed_mask'].squeeze(-1), \
                          output['eval_mask'].squeeze(-1)

y_hat = y_hat[0]
y_true = y_true[0]
eval_mask = eval_mask[0]

error_across_locations = np.abs(y_hat - y_true) * eval_mask
error_across_locations = np.sum(error_across_locations, axis=0) / np.sum(eval_mask, axis=0)

static = pd.read_csv('../data/constant_grid.csv')

fig, ax = plt.subplots(figsize=(7,7))

# Adjust the main plot to make space for the color bar on the right
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Create the scatter plot
scatter = ax.scatter(static['coords.x1'], static['coords.x2'], c=error_across_locations, cmap='Reds', edgecolor='k')

# Create a color bar and place it to the right of the axes
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('MAE')

ax.set_title('MAE Across Locations')

plt.show()

fig.savefig('./writings/figure/error_across_space.png', dpi=300, bbox_inches='tight')

######################################################################


import numpy as np
import matplotlib.pyplot as plt

# Assuming 'static' is a DataFrame with 'LC' and 'POINTID' columns
# and 'error_across_locations' is an array or list of error values
# corresponding to each 'POINTID'.

# error across different land cover types
unique_lc = np.unique(static['LC'].values)
land_cover_types = ['Woody savannas', 'Savannas', 'Grasslands', 'Croplands', 'Urban and Built-up Lands']

# Create a list to hold all errors for the different land cover types
all_errors = []

for lc in unique_lc:
    pointid = static[static['LC'] == lc]['POINTID'].values
    index = pointid - 1
    errors = error_across_locations[index]
    all_errors.append(errors)

# Now plot all boxplots in one figure
fig, ax = plt.subplots(figsize=(7, 4))
# Boxplot of error for each land cover type
bp = ax.boxplot(all_errors)

# Set the x-axis labels to the land cover types, assuming they match the order of unique_lc
ax.set_xticklabels(land_cover_types, size=8)

# Set y label to be MAE
ax.set_ylabel('MAE')

# Save the figure
plt.savefig('./writings/figure/error_across_land_cover.png', dpi=300, bbox_inches='tight')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()  # Display the plot





######################################################################