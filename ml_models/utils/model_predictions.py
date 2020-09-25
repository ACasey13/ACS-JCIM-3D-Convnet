#import tensorflow as tf
#print('\ntensorflow version: {}\n'.format(tf.VERSION))
#from tensorflow.python import keras

import keras
print('Keras version: {}\n'.format(keras.__version__))
import os
import pandas as pd
from data_gen import GenCreator
import numpy as np
from keras import models
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import cm 
import seaborn as sns
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




# putting all of this in a class started as a good idea....
# but now it looks like in the future this should be broken
# up into little functions
class ModelEval():
    def __init__(self,
                 model_file,
                 data_dir = '../data/',
                 train_dirs=None,
                 val_dirs=None,
                 test_dirs=None,
                 labels_store=None):

        self.data_dir = data_dir
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.labels_store = labels_store

        try:
            self.model = models.load_model(model_file)
        except Exception as e:
            print("Could not load model '{}' in class 'ModelEval'".format(model_file))
            print(e)


    def act_vs_pred(self,
                    save_dir,
                    output_label=['gp2'],
                    train_dirs=None,
                    val_dirs=None,
                    test_dirs=None,
                    labels_store=None,
                    augment=True,
                    parent_dir = None,
                    meta={'cube_scale':None,
                          'pot_scale':None,
                          'cube_clip':None,
                          'pot_clip':None}):

        if train_dirs is None: train_dirs = self.train_dirs
        if val_dirs is None: val_dirs = self.val_dirs
        if test_dirs is None: test_dirs = self.test_dirs
        if labels_store is None: labels_store = self.labels_store
        if parent_dir is None: parent_dir = self.data_dir

            
        # get number of plots to produce
        meta_dirs = [train_dirs, val_dirs, test_dirs]
        provided_dirs = [(ix,x) for ix,x in enumerate(meta_dirs) if x is not None]
        n_plots = len(provided_dirs) 
        names = ('train', 'validation', 'test')       

        """
        # get scaling factors
        f = open(os.path.join(self.data_dir,'scale.txt'))
        c_mi = float(f.readline())
        c_ma = float(f.readline())
        p_mi = float(f.readline())
        p_ma = float(f.readline())
        f.close()
        """

        if meta['cube_scale'] is None:
            scale_cube = 1./c_ma
        else:
            scale_cube = meta['cube_scale']
        if meta['pot_scale'] is None:
            scale_pot = 1./p_ma
        else:
            scale_pot = meta['pot_scale']
        if meta['cube_clip'] is None:
            cube_clip=None
        else:
            cube_clip = meta['cube_clip']
        if meta['pot_clip'] is None:
            pot_clip = None
        else:
            pot_clip = meta['pot_clip']

        def get_preds(dirs_file):
            gen = GenCreator(batch_size=64,
                             augment=augment,
                             shuffle=False,    #!!! This must be false for predictions
                             dirs_file=dirs_file,
                             scale_cube=scale_cube,
                             scale_pot=scale_pot,
                             cube_clip=cube_clip,
                             pot_clip=pot_clip,
                             verbose=1,
                             dshape='cube',
                             parent_dir=parent_dir,
                             name=dirs_file,
                             labels_store=labels_store)
            print('moving to predictions...')
            preds = self.model.predict_generator(gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count())

            print('preds shape: {}'.format(preds.shape))

            if augment:
                preds = preds.reshape((24, int(len(preds)/24)))
                preds_mean = preds.mean(axis=0).reshape((-1,1))
                preds_std = preds.std(axis=0).reshape((-1,1))
            else:
                preds = preds.reshape((1, -1))
                preds_mean = preds.mean(axis=0).reshape((-1,1))
                preds_std = preds.std(axis=0).reshape((-1,1))
         

            print('preds shape after average: {}'.format(preds_mean.shape))
            return preds, preds_mean, preds_std

        def get_labels(dirs_file, output_label):
            acts = self._get_labels(dirs_file, labels_store,
                                    out=output_label)
            return acts

        for i, (ix, dirs_file) in enumerate(provided_dirs):
            print('working on directory list: {}'.format(dirs_file))
            X = np.loadtxt(dirs_file, dtype=np.str).flatten()
            y_preds, y_mean, y_std = get_preds(dirs_file)
            y_mean = y_mean.flatten()
            y_std = y_std.flatten()
            y_act = get_labels(dirs_file, output_label).flatten()

            pd_dict = {'molecule': X, 'actual': y_act, 'predicted': y_mean, 
                       'std': y_std}
            pd_orient = {'orientation_{:02d}'.format(i):vals.flatten() for i,vals in enumerate(y_preds)}
            pd_dict.update(pd_orient)
            
            df = pd.DataFrame(pd_dict)
            save_file = os.path.join(save_dir, '{}.h5'.format(names[ix]))
            df.to_hdf(save_file, 'results')
            print('saved result to {}!'.format(save_file))


    @staticmethod
    def _get_labels(dirs_file, labels_store, out=['gp2']):
        df = pd.read_hdf(labels_store, 'dfstore') 
        labels_array = []
        dirs = np.loadtxt(dirs_file, dtype=np.str)
        for dir_name in dirs:
            df_idx = int(dir_name[2:])
            labels_array.append(df[out].loc[df_idx])

        labels_array = np.array(labels_array).reshape((-1,1))
        return labels_array

