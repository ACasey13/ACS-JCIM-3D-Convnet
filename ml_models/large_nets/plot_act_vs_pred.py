import matplotlib as mpl
print('default backend: {}'.format(mpl.get_backend()))
#mpl.rcParams['backend'] = 'GTKAgg'
#mpl.use('TkCairo')

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import os
import scipy.stats as stats
import matplotlib as mpl

print('now using backend: {}'.format(mpl.get_backend()))

def make_plot(plot_dir,
              plot_type='mpl', #can be 'mpl' or 'sns' or 'both'
              prop=None,
              moniker=None,
              train_data=None,
              val_data=None,
              test_data=None):
    if train_data is None and val_data is None and test_data is None:
        raise ValueError('need to provide at least one set of data')
    
    units = {'Dipole Moment': 'Debye', 'Det. Vel.': 'km/s', 'Det. Pres.': 'GPa',
             'Density': 'g/cc', 'Heat of Formation': 'kcal/mol', 'HOMO-LUMO Gap': 'eV',
             'Energy': 'Hartree', 'CJ Temperature': 'K'}

    meta_files = [train_data, val_data, test_data]
    provided_files = [(ix,x) for ix,x in enumerate(meta_files) if x is not None]
    n_plots = len(provided_files)
    names = ['train', 'validate', 'test']
    """
    def make_sns_subplot(df, ix):
            with sns.axes_style('white'):
                g = sns.jointplot('actual', 'predicted', df, kind='scatter', alpha=.2,
                               space=0, zorder=2)
                ax = g.ax_joint
                ax.set_xlabel("Reference %s (%s)" % (prop, units[prop]), size=16)
                ax.set_ylabel("Predicted %s (%s)" % (prop, units[prop]), size=16)

                axmx = g.ax_marg_x
                axmy = g.ax_marg_y
                axmx.spines['bottom'].set_visible(False)
                axmy.spines['left'].set_visible(False)
                
                ax.spines['left'].set_position(('outward', 4))
                ax.spines['right'].set_position(('outward', 4))

                sns.set(font_scale=1.5)
                x0, x1 = g.ax_joint.get_xlim()
                y0, y1 = g.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                print('setting lims to {}'.format(lims))
                g.ax_joint.plot(lims, lims,':k', zorder=9)
                rmse = sqrt(mean_squared_error(df['actual'],df['predicted']))
                r2 = r2_score(df['actual'], df['predicted'])

                ax.text(0,.85,'$R^2$: {:.3f}\nRMSE: {:5.3f}'.format(r2, rmse),
                        fontsize=16, transform = ax.transAxes)
            #g.fig.suptitle()

            g.savefig(os.path.join(plot_dir, '{}_sns.png'.format(names[ix])))
            print('subplot {}.png finished!'.format(names[ix]))
    """
    # you can't combine sns jointplots as subfigures so this 
    # takes their separate images and concatenates them
    # the above code was changed, so this needs to be updated to work
    """
    for i, (ix, data_file) in enumerate(provided_files):
        df = pd.read_hdf(data_file, 'results')
        make_subplot(df, ix)
    # append figures
    images = []
    for i, (ix, data_file) in enumerate(provided_files):
        images.append(Image.open(os.path.join(plot_dir, 'tmp_{}.png'.format(ix))))
    widths, heights = zip(*(i.size for i in images))

    width = sum(widths)
    height = heights[0]

    new_im = Image.new('RGB', (width, height))
    x_offset = 0
    for image in images:
        new_im.paste(image, (x_offset, 0))
        x_offset += image.size[0]
    plot_location = os.path.join(plot_dir, plot_name)
    new_im.save(plot_location)
    """

    def make_mpl_plot(df, ix):
        actual = df['actual']
        predicted = df['predicted']
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        plt.tick_params(labelsize=8)
        ax.set_aspect('equal')
        ax.grid(alpha=0.8, linestyle='-', zorder=0)
        ax.set_axisbelow(True)
        hb = ax.hexbin(actual, predicted, cmap='viridis', mincnt=1,
                       gridsize=100, bins=None, extent=[actual.min(),
                       actual.max(), actual.min(), actual.max()],
                       linewidths=0.001, zorder=9)
        #ax.plot(min(a.min(axis=0), b.min(axis=0)), max(a.max(axis=0), b.max(axis=0)),'k-')
        line = mlines.Line2D([0, 1], [0, 1], color='red', zorder=5)
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        #ax.set_title("Hexagon binning of %s" % prop, fontsize=11)
        cb = fig.colorbar(hb, ax=ax)
        cb.ax.tick_params(labelsize=8)
        #cb.set_label('log10(N)', fontsize=16)
        cb.set_label('Counts', fontsize=8)
        rmse = sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        textstr = 'RMSE$=%1.3G$ %s\nR$^2=%1.3G$' % (rmse, units[prop], r2)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.10, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        ax.set_xlabel("Ref %s (%s)" % (prop, units[prop]), size=8)
        ax.set_ylabel("Pred %s (%s)" % (prop, units[prop]), size=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, '{}_{}_mpl.eps'.format(moniker, names[ix])))
        print('subplot {}_{}_mpl.pdf finished!'.format(moniker, names[ix]))


    for ix, data_file in (provided_files):
        df = pd.read_hdf(data_file, moniker)
        if plot_type == 'sns':
            make_sns_subplot(df, ix)
        elif plot_type == 'mpl':
            make_mpl_plot(df,ix)
        elif plot_type == 'both':
            make_sns_subplot(df,ix)
            make_mpl_plot(df,ix)
        else:
            raise ValueError('plot type not understood!')


if __name__ == '__main__':

    plot_dir = './vanilla_multi_2_cont_3'
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

