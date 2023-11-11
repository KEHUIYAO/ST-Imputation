import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsl.utils import numpy_metrics
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


log_dir = 'log/soil_moisture_sparse_point/st_transformer/20231103T174145_0/in_sample.npz'

# log_dir = 'log/soil_moisture_hb_point/st_transformer/20231110T063243_0/in_sample.npz'

output = np.load(log_dir)

y_hat, y_true, observed_mask, eval_mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['observed_mask'].squeeze(-1), \
                          output['eval_mask'].squeeze(-1)



check_mae = numpy_metrics.masked_mae(y_hat, y_true, eval_mask)

n_eval = np.sum(eval_mask)
print(f'Evalpoint: {n_eval}')
print(f'Test MAE: {check_mae:.5f}')
#
# # in-situ data
# df = pd.read_csv('../data/Insitu_gap_filling_data.csv')
# df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
# y2 = df[df['Date'].dt.year == 2016].copy()
# y2 = y2.pivot(index='Date', columns='POINTID', values='SMAP_1km').values
# y2 = y2[np.newaxis, ...]
# mask = np.ones_like(y2)
# mask[np.isnan(y2)] = 0
#
# # calculate ubrmse
# bias = np.mean((y2 - y_hat)[mask == 1])
# tmp = (y2-y_hat)**2
# tmp = np.mean(tmp[mask==1])
# ubrmse = np.sqrt(tmp - bias**2)
# print(f'UBRMSE: {ubrmse:.5f}')
#
# # calculate correlation
# corr = np.corrcoef(y2[mask == 1], y_hat[mask == 1])[0, 1]
# print(f'Correlation: {corr:.5f}')



all_target_np = output['y'].squeeze(-1)
all_evalpoint_np = output['eval_mask'].squeeze(-1)
all_observed_np = output['observed_mask'].squeeze(-1)

# print how many evalpoints we have
print(f'Evalpoint: {np.sum(all_evalpoint_np)}')

if 'imputed_samples' in output:
    samples = output['imputed_samples']
    samples = samples.squeeze(-1)
else:
    samples = output['y_hat']
    samples = samples.squeeze(-1)[:, np.newaxis, ...]

qlist =[0.05,0.25,0.5,0.75,0.95]
quantiles_imp= []
for q in qlist:
    tmp = np.quantile(samples, q, axis=1)
    quantiles_imp.append(tmp*(1-all_observed_np) + all_target_np * all_observed_np)
    # quantiles_imp.append(tmp)


L = all_target_np.shape[1]
K = all_target_np.shape[2]

dataind = 0

# #######################################
# plt.rcParams["font.size"] = 16
# fig, axes = plt.subplots(nrows=9, ncols=4,figsize=(36, 24.0))
#
# fig.delaxes(axes[-1][-1])
#
# offset = 0
# for k in range(offset, 36+offset):
#     df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
#     df = df[df.y != 0]
#     df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_observed_np[dataind,:,k]})
#     df2 = df2[df2.y != 0]
#     row = (k-offset) // 4
#     col = (k-offset) % 4
#     axes[row][col].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CSDI')
#     axes[row][col].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
#                     color='g', alpha=0.3)
#     # axes[row][col].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
#     axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
#     if col == 0:
#         plt.setp(axes[row, 0], ylabel='value')
#     if row == -1:
#         plt.setp(axes[-1, col], xlabel='time')
#
#
# plt.show()
# #######################################




# ###############################
# plt.rcParams["font.size"] = 16
# fig, axes = plt.subplots(figsize=(24, 12))
# k = 0
# # date_range is 2016-01-01 to 2020-12-31
# start_date = pd.to_datetime('2016-01-01')
# end_date = pd.to_datetime('2019-12-31')
# date_range = pd.date_range(start=start_date, end=end_date)
# df = pd.DataFrame({"x": date_range, "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
# df = df[df.y != 0]
# df2 = pd.DataFrame({"x":date_range, "val":all_target_np[dataind,:,k], "y":all_observed_np[dataind,:,k]})
# df2 = df2[df2.y != 0]
#
# axes.plot(date_range, quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid')
#
# axes.plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None', markersize=10)
# axes.plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None', markersize=10)
#
# axes.tick_params(axis='both', labelsize=30)
#
# axes.set_ylabel('value', fontsize=30)
# axes.set_xlabel('time', fontsize=30)
#
# # # Set major ticks to show the first day of every month
# # axes.xaxis.set_major_locator(mdates.MonthLocator())
# # # Format major ticks as 'YYYY-MM'
# # axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#
#
# # Set major ticks to show the year
# axes.xaxis.set_major_locator(mdates.YearLocator())
# axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#
#
# axes.tick_params(axis='both', labelsize=20)
#
# # Rotate and align the x labels
# fig.autofmt_xdate()
#
#
# axes.set_xlim(start_date, end_date)
#
#
# # Create custom legend
# legend_elements = [Line2D([0], [0], color='g', lw=4, label='imputed'),
#                    Line2D([0], [0], color='b', marker='o', lw=0, markersize=10, label='validation'),
#                    Line2D([0], [0], color='r', marker='x', lw=0, markersize=10, label='observed')]
#
# axes.legend(handles=legend_elements, loc='upper right', fontsize=20)
#
#
# plt.show()
#
# # fig.savefig('../writings/figure/smap_1km.png', dpi=300)
# fig.savefig('../writings/figure/smap_hydroblocks.png', dpi=300)
# ###############################


###############################
y_hat = y_hat.reshape(y_hat.shape[1], 36, 36)
y_true = y_true.reshape(y_true.shape[1], 36, 36)
start_date = pd.to_datetime('2016-01-01')
end_date = pd.to_datetime('2019-12-31')
dates = pd.date_range(start='2016-01-01', end='2019-12-31')

# Set the number of time steps
time_steps = 14

# Create a figure with subplots - 2 rows for y_hat, y_true
fig, axes = plt.subplots(2, time_steps, figsize=(10, 3))  # Adjust the figsize as needed
cmap = 'YlGn'
offset = 202
# Plot each time step for y_true
for i in range(time_steps):
    ax = axes[0, i]
    ax.imshow(y_true[i+offset, :, :], cmap=cmap)
    ax.axis('off')
    # set title to the date
    ax.set_title(dates[i+offset].strftime('%m-%d'))

# Plot each time step for y_hat
for i in range(time_steps):
    ax = axes[1, i]
    ax.imshow(y_hat[i+offset, :, :], cmap=cmap)
    ax.axis('off')


# Optionally, you can set common titles for each row
axes[0, 0].set_ylabel('True')
axes[1, 0].set_ylabel('Predicted')


plt.tight_layout()
plt.show()

fig.savefig('../writings/figure/missing_at_time_points_smap.png', dpi=300)
###############################

