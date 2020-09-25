import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

def make_plot(plot_dir,
              prop=None,
              moniker=None,
              train_data=None,
              val_data=None,
              test_data=None):

    if train_data is None and val_data is None and test_data is None:
        raise ValueError('need to provide at least one set of data')

    units = {'Dipole Moment': 'Debye', 'Det. Vel.': 'km/s', 'Det. Pres.': 'GPa',
             'Density': 'g/cc', 'Heat of Formation': 'kcal/mol', 'HOMO-LUMO Gap': 'eV',
             'Energy': 'Hartree', 'CJ Temperature': '$^{\circ}$C'}

    meta_files = [train_data, val_data, test_data]
    provided_files = [(ix,x) for ix,x in enumerate(meta_files) if x is not None]
    n_plots = len(provided_files)
    names = ['train', 'validate', 'test']
    var_s = ['orientation_{:02d}'.format(i) for i in range(24)]

    percent = 7.5

    if not os.path.exists(os.path.join(plot_dir, 'kde')):
        os.mkdir(os.path.join(plot_dir, 'kde'))

    for ix, data_file in provided_files:
        df = pd.read_hdf(data_file, moniker)
        #print(list(df.columns))
        df_reduced = df.drop(['molecule', 'std', 'actual', 'predicted'], axis=1)
        arr = df_reduced.values
        median = np.median(arr, axis=1)
        #print(median.shape)
        sort = np.sort(arr, axis=1)
        sort = sort[:, 2:-2].mean(axis=1)
        act_mean = df['actual'].mean()
        mi = -percent/100*act_mean 
        ma =  percent/100*act_mean 
        #bw = (ma - mi) /  4
        bw = None
        x = np.linspace(mi, ma, 250)
        print('creating kde instances')
        kdes = [stats.gaussian_kde(df[var] - df['actual'], bw) for var in var_s]
        fig, ax = plt.subplots(figsize=(6,6))
        for n, kde in enumerate(kdes):
            ax.plot(x, kde(x))#, label=var_s[n])
        ax.plot(x, stats.gaussian_kde(df['predicted'] - df['actual'], bw)(x), '-k', label='avg')
        ax.plot(x, stats.gaussian_kde(median - df['actual'], bw)(x), '--k', label='med')
        ax.plot(x, stats.gaussian_kde(sort - df['actual'], bw)(x), ':k', label='trim')
        ax.legend(edgecolor='w')
        ax.set_xlabel('Residuals ({})'.format(units[prop]))
        ax.set_ylabel('KDE Estimate')
        plt.savefig(os.path.join(plot_dir, 'kde', '{}_{}_kde.pdf'.format(moniker, names[ix])))
        print('plot {}_{}_kde.pdf finished!'.format(moniker, names[ix]))
        plt.close('all')


if __name__ == '__main__':

    plot_dir = './vanilla_multi_2'
    train_data = os.path.join(plot_dir, 'train.h5')
    test_data = os.path.join(plot_dir, 'test.h5')

    out_list = ['density', 'detvel', 'detpres',
                'gp2', 'energy', 'heat',
                'cjtemp', 'gp1']

    moniker_to_formal = {'density': 'Density',
                         'detvel': 'Det. Vel.',
                         'detpres': 'Det. Pres.',
                         'gp2': 'Dipole Moment',
                         'energy': 'Energy',
                         'heat': 'Heat of Formation',
                         'cjtemp': 'CJ Temperature',
                         'gp1': 'HOMO-LUMO Gap'}

    for p in out_list:
        make_plot(plot_dir,
                  prop=moniker_to_formal[p],
                  moniker=p,
                  train_data=train_data,
                  test_data=test_data)

