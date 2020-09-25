import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

    if not os.path.exists(os.path.join(plot_dir, 'pairplot')):
        os.mkdir(os.path.join(plot_dir, 'pairplot'))

    for ix, data_file in provided_files:
        df = pd.read_hdf(data_file, moniker)
        print('read in {} : {}'.format(names[ix], moniker))
        grid = sns.PairGrid(data = df, vars=var_s)
        print('pairgrid instantiated')
        grid = grid.map_upper(sns.kdeplot)
        print('upper mapped')
        grid = grid.map_diag(plt.hist, bins=100)
        print('diag mapped')
        print('saving plot...')
        grid.savefig(os.path.join(plot_dir, 'pairplot','{}_{}_pairplot.pdf'.format(moniker, names[ix])))
        print('subplot {}_{}_pairplot.pdf finished!'.format(moniker, names[ix]))


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

