import pandas as pd
import numpy as np

df_file = '../data/cands-concat-clean.h5'

cols_to_norm = ['density', 'heat', 'exenergy', 'detvel', 'detpres', 'cjtemp', 'gp1', 'gp2', 'gp3', 'energy']

df = pd.read_hdf(df_file, 'dfstore')

means = []
stds = []
idxs = df.index[df['status'] == 5]

for col in cols_to_norm:
    mean = df.loc[idxs, col].mean()
    std = df.loc[idxs, col].std()
    df.loc[idxs, col] = (df.loc[idxs, col]-mean)/std
    means.append(mean)
    stds.append(std)
    print('working on column: {}'.format(col))
    print('mean: {}, std: {}\n'.format(mean, std))

stats = np.array([means, stds])

df_stat = pd.DataFrame(stats, columns=cols_to_norm, index=['mean', 'std'])

df_stat.to_hdf('../data/cands-concat-clean-norm_params.h5', 'dfstore', format='table')
np.savetxt('../data/cands-concat-clean-norm_params.txt', stats)
df.to_hdf('../data/cands-concat-clean-norm.h5', 'dfstore', format='table')



