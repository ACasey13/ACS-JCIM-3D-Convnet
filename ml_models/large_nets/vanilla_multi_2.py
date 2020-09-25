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
from data_gen_multi import GenCreatorMulti
from get_scale import get_scale
from my_callbacks import LossHistory
from keras.callbacks import ReduceLROnPlateau
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model_modules import vanilla_multi_2 as vanilla
from keras.utils import plot_model

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

# set up model
model = vanilla()

print('Model Summary:')
model.summary()

# define some filepaths and hyper parameters
BATCH_SIZE = 64
N_EPOCHS = 50
data_dir = os.path.abspath('../data')
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

# print graph of model
#plot_model(model, to_file='model.pdf')
#print('finished plotting model!!!')

# get needed files from data directory
train_dirs_file = os.path.join(data_dir, 'train_dirs.txt')
test_dirs_file = os.path.join(data_dir, 'test_dirs.txt')

cube_clip = .16
pot_clip = .6

# !!!!!!!!!!!!!!!!!!!!!!!!!
#rewriting for this run....
scale_cube = 1./cube_clip
scale_pot = 1./pot_clip

# instantiate the data flow generators
print('\nInstantiating data flow generators...')
train_gen = GenCreatorMulti(BATCH_SIZE, augment=True, dirs_file=train_dirs_file,
                       scale_cube=scale_cube, scale_pot=scale_pot,
                       pot_clip=pot_clip, cube_clip=cube_clip,
                       name='train', verbose=0, dshape='cube',
                       labels_store=df_store)

test_gen = GenCreatorMulti(BATCH_SIZE, augment=False, dirs_file=test_dirs_file,
                      scale_cube=scale_cube, scale_pot=scale_pot,
                      pot_clip=pot_clip, cube_clip=cube_clip,
                      name='test', verbose=0, dshape='cube',
                      labels_store=df_store)

# record loss and model weights throughout training
#l_hist = LossHistory(model_dir)
l_hist = keras.callbacks.CSVLogger(os.path.join(model_dir, 'log.csv'), separator=',', append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=0.00001, verbose=1)
tb = keras.callbacks.TensorBoard(model_dir, batch_size=BATCH_SIZE, write_grads=False,
                                 write_images=False, write_graph=True,  update_freq=1000,
                                 histogram_freq=0)
filepath = os.path.join(model_dir, 'model.{epoch:02d}.hdf5')
chk = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                      save_best_only=False, save_weights_only=False,
                                      mode='auto', period=1)

# fit the model
print('\nFitting model....')
history = model.fit_generator(train_gen,
                              verbose=1,
                              epochs=N_EPOCHS,
                              validation_data=test_gen,
                              use_multiprocessing=True,
                              workers=mp.cpu_count(),
                              callbacks=[l_hist, chk, reduce_lr])




