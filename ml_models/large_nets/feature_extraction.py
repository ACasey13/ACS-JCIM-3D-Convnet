print('Running file {}'.format(__file__))
print('-' * 50 + '\n')

#import tensorflow as tf
#print('\ntensorflow version: {}\n'.format(tf.VERSION))
#from tensorflow.python import keras
import keras
print('Keras version: {}\n'.format(keras.__version__))
import sys
import os
#append utils directory to PYTHONPATH if not done so already
try: 
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

if os.path.abspath('../utils') not in [os.path.abspath(path) for path in user_paths]:
    sys.path.append(os.path.abspath('../utils/'))

import json
import multiprocessing as mp
from data_gen import GenCreator
from get_scale import get_scale
from my_callbacks import LossHistory
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model_modules import small_head

# uncomment the following line to suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####################################################################
####################################################################

# name this model
# must keep '.py' (due to how it is called later...)
model_name = __file__
print('')
print('-'*50)
print('Model name: {}'.format(model_name[:-3]))
print('-'*50)
print('')

# load last model
model = models.load_model('../large_nets/vanilla_multi_2_cont_3/model.09.hdf5')

print('Original Model Summary:')
model.summary()

headless_model = models.Model(inputs=model.input,
                              outputs=model.get_layer('dense_3').output)

#flat = layers.Flatten(name='final_flatten')(headless_model.output)
#new_model = models.Model(inputs=headless_model.input,
#                         outputs=flat)

print('Updated model for fine tuning:')
headless_model.summary()


# define some filepaths and hyper parameters
BATCH_SIZE = 64
N_EPOCHS = 50
data_dir = os.path.abspath('../data')
parent_dir = os.path.abspath('../data/molecules')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, model_name[:-3])
df_store = os.path.join(data_dir, '../data/cands-concat-clean.h5')

# create new directory for this model
# model information will be saved here
print('\nChanging directory to {}.'.format(model_dir))
if os.path.exists(model_dir):
    os.chdir(model_dir)
else:
    os.mkdir(model_dir)
    os.chdir(model_dir)

# get needed files from data directory
full_dirs_file = os.path.join(data_dir, 'molecule_list.txt')
train_dirs_file = os.path.join(data_dir, 'train_dirs.txt')
test_dirs_file = os.path.join(data_dir, 'test_dirs.txt')
#scale_file = os.path.join(data_dir, 'scale.txt')

# read max .cube and .pot values from training files
# store these values to '../data/scale.txt' for easy
# read-in in subsequent runs

"""
print('\nGetting cube scaling factor...')
if os.path.isfile(scale_file):
    f = open(scale_file)
    c_mi = float(f.readline())
    c_ma = float(f.readline())
    p_mi = float(f.readline())
    p_ma = float(f.readline())
    cube_scale = [c_mi, c_ma]
    pot_scale = [p_mi, p_ma]
    f.close()
else:
    cube_scale = get_scale(train_dirs_file, 'cube')
    pot_scale = get_scale(train_dirs_file, 'pot')
    print("\nwriting scale info to '{}'...".format(scale_file))
    sf = open(scale_file, 'w')
    sf.write(str(cube_scale[0])+'\n')
    sf.write(str(cube_scale[1])+'\n')
    sf.write(str(pot_scale[0])+'\n')
    sf.write(str(pot_scale[1])+'\n')
    sf.close()

# right now, the code preprocesses the data by dividing my the 
# maximum value found in the training set
# this can be changed here by setting 'scale_cube' and 
# 'scale_pot' to whatever value you want to normalize to
print('cube scale: {}'.format(cube_scale))
scale_cube = 1./cube_scale[1]
print('pot scale: {}'.format(pot_scale))
scale_pot = 1./pot_scale[1]
"""
cube_clip = .16
pot_clip = .6

# !!!!!!!!!!!!!!!!!!!!!!!!!
#rewriting for this run....
scale_cube = 1./cube_clip
scale_pot = 1./pot_clip

#print('train_dirs_file: {}'.format(train_dirs_file))
#print('test_dirs_file: {}'.format(test_dirs_file))
print('will be grabbing electronic structure from {}'.format(parent_dir))

# instantiate the data flow generators
print('\nInstantiating data flow generators...')
full_gen = GenCreator(BATCH_SIZE, augment=True, shuffle=False, dirs_file=full_dirs_file,
                       parent_dir=parent_dir, #this must be present!!!
                       scale_cube=scale_cube, scale_pot=scale_pot,
                       pot_clip=pot_clip, cube_clip=cube_clip,
                       name='full', verbose=0, dshape='cube',
                       labels_store=df_store)

"""
train_gen = GenCreator(BATCH_SIZE, augment=True, shuffle=False, dirs_file=train_dirs_file,
                       parent_dir=parent_dir, #this must be present!!!
                       scale_cube=scale_cube, scale_pot=scale_pot,
                       pot_clip=pot_clip, cube_clip=cube_clip,
                       name='train', verbose=0, dshape='cube',
                       labels_store=df_store, out='h50')

test_gen = GenCreator(BATCH_SIZE, augment=True, shuffle=False, dirs_file=test_dirs_file,
                       parent_dir=parent_dir, #this must be present!!!
                       scale_cube=scale_cube, scale_pot=scale_pot,
                       pot_clip=pot_clip, cube_clip=cube_clip,
                       name='test', verbose=0, dshape='cube',
                       labels_store=df_store, out='h50')
"""

# fit the model
print('\nmaking full-set predictions....')
full_preds = headless_model.predict_generator(full_gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count())


"""
print('\nmaking train predictions....')
train_preds = headless_model.predict_generator(train_gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count())

print('\nmaking test predictions....')
test_preds = headless_model.predict_generator(test_gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count())

"""
np.save('full_preds.npy', full_preds)
#np.save('train_preds.npy', train_preds)
#np.save('test_preds.npy', test_preds)
print('Files saved! Program successfull!')


"""
# upon model completion, save everything
print("\nSaving model as '{}.h5'...".format(model_name[:-3]))
new_model.save('{}.h5'.format(model_name[:-3]))
print('Model saved!')

print("\nSaving 'history.history' to '{}.json'".format(model_name[:-3]))
with open('{}.json'.format(model_name[:-3]), 'w') as f:
    json.dump(history.history, f)

"""


# most of the code below has been moved to separate files 
# in order to separate the fitting and evaluation processes
"""
train_mae = history.history['mean_absolute_error']
test_mae = history.history['val_mean_absolute_error']
#train_mse = history.history['mean_squared_error']
#test_mse = history.history['val_mean_squared_error']
train_loss = history.history['loss']
test_loss = history.history['val_loss']
epochs = range(1, len(train_mae)+1)


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,12))
ax1.plot(epochs, train_loss, label='train')
ax1.plot(epochs, test_loss, label='test')
ax1.set_title('MSE Loss')
ax1.legend(edgecolor='w')

ax2.plot(epochs, train_mae, label='train')
ax2.plot(epochs, test_mae, label='test')
ax2.set_title('MAE')
ax2.legend(edgecolor='w')
plt.savefig('model_history.png')

print("\nModel History saved to 'model_history.png'")

print('\nSuccessfully fit model and plotted model history.')

print('\nMoving on to model predictions...')

print('\nRe-instantiating data flow generators...')
test_gen = GenCreator(10, augment=False, dirs_file=test_dirs_file,
                      scale_cube=scale_cube, scale_pot=scale_pot,
                      verbose=0, name='test', labels_store=df_store)

train_gen = GenCreator(10, augment=False, dirs_file=train_dirs_file,
                       scale_cube=scale_cube, scale_pot=scale_pot,
                       verbose=0, name='train', labels_store=df_store)

print('\nMaking test predictions....')
preds_test = model.predict_generator(test_gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count()).flatten()
print('\nMaking train predictions....')
preds_train = model.predict_generator(train_gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count()).flatten()


print('\npreds test shape: {}'.format(preds_test.shape))
print('preds train shape: {}'.format(preds_train.shape))

def get_labels(test_dirs_file, out='gp2'):
    df = pd.read_hdf(df_store, 'dfstore')
    labels_array = []
    dirs = np.loadtxt(test_dirs_file, dtype=np.str)
    for dir_name in dirs:
        df_idx = int(dir_name[2:])
        labels_array.append(df[out].iloc[df_idx])

    labels_array = np.array(labels_array)
    return labels_array

print('\ngetting test labels...')
acts_test = get_labels(test_dirs_file)
print('\ngetting train labels...')
acts_train = get_labels(train_dirs_file)

print('acts test shape: {}'.format(acts_test.shape))
print('acts train shape: {}'.format(acts_train.shape))

print('\nProducing pred vs act plot...')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
ax1.hexbin(acts_train, preds_train, gridsize=100, cmap='Blues')
ax2.hexbin(acts_test, preds_test, gridsize=100, cmap='Reds', label='test')
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
mi_lim = min(xlim[0], ylim[0])
ma_lim = max(xlim[1], ylim[1])
nlim = (mi_lim, ma_lim)
x_plot = np.linspace(mi_lim, ma_lim, 250)
ax1.plot(x_plot, x_plot, '-r')
ax1.set_xlim(nlim)
ax1.set_ylim(nlim)
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
mi_lim = min(xlim[0], ylim[0])
ma_lim = max(xlim[1], ylim[1])
nlim = (mi_lim, ma_lim)
x_plot = np.linspace(mi_lim, ma_lim, 250)
ax2.plot(x_plot, x_plot, '-r')
ax2.set_xlim(nlim)
ax2.set_ylim(nlim)
ax1.set_title('Train')
ax2.set_title('Test')

print("Saving figure to 'pred_plot.png'...")
plt.savefig('pred_plot.png')

"""
