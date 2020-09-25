import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from plot_settings import *

models_base = os.path.abspath('./')
model_dirs = ['./vanilla_multi_2_elu_cont_1']

def create_plot(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'history')):
        os.mkdir(os.path.join(model_dir, 'history'))
    network_name = os.path.split(model_dir)[1]
    loss_file = os.path.join(model_dir, 'log.csv')

    hist = pd.read_csv(loss_file)
    epochs = hist.index + 1

    #get property names from df
    cols = list(hist.columns)
    props = [col[:-5] for col in cols if col[-4:]=='loss' and col[:3] != 'val' and col != 'loss']

    for prop in props:
        y1 = hist[prop + '_loss']
        y2 = hist['val_' + prop + '_loss']
        y3 = hist[prop + '_mean_absolute_error']
        y4 = hist['val_' + prop + '_mean_absolute_error'] 

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.plot(epochs, y1, label='train')
        ax1.plot(epochs, y2, label='test')
        xlim = ax1.get_xlim()
        ax1.set_xlim(1, xlim[1])
    
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ylim = ax1.get_ylim()
        ax1.set_ylim(ylim[0], min(ylim[1], y2[2]))

        ax1.legend(edgecolor='w', loc='upper right')
        plt_format(ax1)

        ax2.plot(epochs, y3, label='train')
        ax2.plot(epochs, y4, label='test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_xlim(1, xlim[1])
        ylim = ax2.get_ylim()
        ax2.set_ylim(ylim[0], min(ylim[1], y4[2]))

        plt_format(ax2)
        plt.savefig(os.path.join(model_dir, 'history', '{}_history.pdf'.format(prop)))

for model_dir in model_dirs:
    print('working on plot {}'.format(model_dir))
    create_plot(model_dir)
