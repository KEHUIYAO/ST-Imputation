import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsl.utils import numpy_metrics


log_dir = 'log/soil_moisture_sparse_point/interpolation/20230918T164007_727343612/output.npz'

log_dir = 'log/soil_moisture_sparse_point/spin_h/20230919T034448_769252367/output.npz'

log_dir = 'log/soil_moisture_sparse_point/grin/20230921T031017_968938493/output.npz'

# log_dir = 'log/soil_moisture_sparse_point/csdi/20230921T025841_774017230/output.npz'
#
# log_dir = 'log/soil_moisture_sparse_point/interpolation/20230920T224810_88924231/output.npz'
#
# log_dir = 'log/soil_moisture_sparse_point/spin_h/20230921T041518_997739772/output.npz'

log_dir = 'log/soil_moisture_sparse_point/interpolation/20230921T131602_38016970/output.npz'

log_dir = 'log/soil_moisture_sparse_point/csdi/20230921T214251_284367313/output.npz'



output = np.load(log_dir)

y_hat, y_true, observed_mask, eval_mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['observed_mask'].squeeze(-1), \
                          output['eval_mask'].squeeze(-1)

check_mae = numpy_metrics.masked_mae(y_hat, y_true, eval_mask)

n_eval = np.sum(eval_mask)
print(f'Evalpoint: {n_eval}')
print(f'Test MAE: {check_mae:.5f}')

# in-situ data
df = pd.read_csv('../data/Insitu_gap_filling_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
y2 = df[df['Date'].dt.year == 2016].copy()
y2 = y2.pivot(index='Date', columns='POINTID', values='SMAP_1km').values
y2 = y2[np.newaxis, ...]
mask = np.ones_like(y2)
mask[np.isnan(y2)] = 0

# calculate ubrmse
bias = np.mean((y2 - y_hat)[mask == 1])
tmp = (y2-y_hat)**2
tmp = np.mean(tmp[mask==1])
ubrmse = np.sqrt(tmp - bias**2)
print(f'UBRMSE: {ubrmse:.5f}')

# calculate correlation
corr = np.corrcoef(y2[mask == 1], y_hat[mask == 1])[0, 1]
print(f'Correlation: {corr:.5f}')



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

plt.rcParams["font.size"] = 16
fig, axes = plt.subplots(nrows=9, ncols=4,figsize=(36.0, 24.0))

fig.delaxes(axes[-1][-1])

offset = 0
for k in range(offset, 36+offset):
    df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
    df = df[df.y != 0]
    df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_observed_np[dataind,:,k]})
    df2 = df2[df2.y != 0]
    row = (k-offset) // 4
    col = (k-offset) % 4
    axes[row][col].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CSDI')
    axes[row][col].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
                    color='g', alpha=0.3)
    axes[row][col].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
    axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
    if col == 0:
        plt.setp(axes[row, 0], ylabel='value')
    if row == -1:
        plt.setp(axes[-1, col], xlabel='time')


plt.show()


