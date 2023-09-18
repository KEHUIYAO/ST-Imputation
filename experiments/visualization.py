import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsl.utils import numpy_metrics

# log_dir = 'log/gp_point/diffgrin/20230905T112029_455519236/output.npz'
# log_dir = 'log/gp_point/diffgrin/20230905T121230_345791943/output.npz'
# log_dir = 'log/gp_point/diffgrin/20230905T140644_801483104/output.npz'
# log_dir = 'log/air36/diffgrin/20230906T151301_772579956/output.npz'
# log_dir = 'log/descriptive_point/diffgrin/20230907T160243_801791294/output.npz'

#log_dir = 'log/gp_point/mean/20230904T120758_203730871/output.npz'
#log_dir = 'log/gp_point/interpolation/20230904T133532_834789790/output.npz'
# log_dir = 'log/gp_point/grin/20230908T123714_666920856/output.npz'
# log_dir = 'log/gp_point/diffgrin/20230908T143312_924613142/output.npz'

# log_dir = 'log/air36/diffgrin/20230908T154234_230789540/output.npz'
# log_dir = 'log/air36/grin/20230908T210723_211066964/output.npz'

log_dir = 'log/soil_moisture_point/diffgrin/20230912T163638_202025312/output.npz'

log_dir = 'log/soil_moisture_point/interpolation/20230912T221617_447206417/output.npz'

log_dir = 'log/soil_moisture_point/spin_h/20230913T152648_613511213/output.npz'
log_dir = 'log/soil_moisture_point/grin/20230913T165400_783353967/output.npz'

log_dir = 'log/soil_moisture_point/csdi/20230914T051816_921699308/output.npz'

log_dir = 'log/air36/csdi/20230914T051347_308853642/output.npz'

log_dir = 'log/soil_moisture_sparse_point/interpolation/20230914T164411_237582339/output.npz'

log_dir = 'log/soil_moisture_sparse_point/interpolation/20230915T144759_108021515/output.npz'

log_dir = 'log/soil_moisture_sparse_point/csdi/20230915T055142_757620585/output.npz'

log_dir = 'log/soil_moisture_sparse_point/spin_h/20230917T221847_9760443/output.npz'

# log_dir = 'log/soil_moisture_sparse_point/spin_h/20230917T173058_522756964/output.npz'

log_dir = 'log/soil_moisture_sparse_point/interpolation/20230917T122607_485269053/output.npz'


# log_dir = 'log/soil_moisture_sparse_point/spin_h/20230918T025242_589438871/output.npz'

output = np.load(log_dir)

y_hat, y_true, observed_mask, eval_mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['observed_mask'].squeeze(-1), \
                          output['eval_mask'].squeeze(-1)



check_mae = numpy_metrics.masked_mae(y_hat, y_true, eval_mask)
print(f'Test MAE: {check_mae:.5f}')



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
    # quantiles_imp.append(tmp*(1-all_observed_np) + all_target_np * all_observed_np)
    quantiles_imp.append(tmp)






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


