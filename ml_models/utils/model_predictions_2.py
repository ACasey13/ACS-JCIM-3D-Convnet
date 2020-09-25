#import tensorflow as tf
#print('\ntensorflow version: {}\n'.format(tf.VERSION))
#from tensorflow.python import keras

import keras
print('Keras version: {}\n'.format(keras.__version__))
import os
import pandas as pd
from data_gen_multi import GenCreatorMulti
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
                    out_list = ['density', 'detvel', 'detpres',
                                'gp2', 'energy', 'heat',
                                'cjtemp', 'gp1'],
                    train_dirs=None,
                    val_dirs=None,
                    test_dirs=None,
                    labels_store=None,
                    augment=True,
                    meta={'cube_scale':None,
                          'pot_scale':None,
                          'cube_clip':None,
                          'pot_clip':None}):

        if train_dirs is None: train_dirs = self.train_dirs
        if val_dirs is None: val_dirs = self.val_dirs
        if test_dirs is None: test_dirs = self.test_dirs
        if labels_store is None: labels_store = self.labels_store

            
        # get number of plots to produce
        meta_dirs = [train_dirs, val_dirs, test_dirs]
        provided_dirs = [(ix,x) for ix,x in enumerate(meta_dirs) if x is not None]
        n_plots = len(provided_dirs) 
        names = ('train', 'validation', 'test')       


        """ # get scaling factors
        f = open(os.path.join(self.data_dir,'scale.txt'))
        c_mi = float(f.readline())
        c_ma = float(f.readline())
        p_mi = float(f.readline())
        p_ma = float(f.readline())
        f.close()"""

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
            gen = GenCreatorMulti(batch_size=64,
                             augment=augment,
                             shuffle=False,    #!!! This must be false for predictions
                             dirs_file=dirs_file,
                             scale_cube=scale_cube,
                             scale_pot=scale_pot,
                             cube_clip=cube_clip,
                             pot_clip=pot_clip,
                             verbose=1,
                             dshape='cube',
                             name=dirs_file,
                             labels_store=labels_store)
            print('moving to predictions...')
            preds = self.model.predict_generator(gen,
                              verbose=1,
                              use_multiprocessing=True,
                              workers=mp.cpu_count())

            #print('preds type: {}'.format(type(preds)))
            #print('preds length: {}'.format(len(preds)))
            #print('preds first elem type: {}'.format(type(preds[0])))
            #print('preds first elem length: {}'.format(len(preds[0])))

            formatted_preds = []
            preds_mean = []
            preds_std = []

            
            """if dshape == 'cube':
                div = 24
            elif dshape == 'rect':
                div = 4"""
            div=24

            if augment:
                for pred in preds:
                    pred = pred.reshape((div, int(len(pred)/div)))
                    formatted_preds.append(pred)
                                    
                    preds_mean.append(pred.mean(axis=0).reshape((-1,1)))
                    preds_std.append(pred.std(axis=0).reshape((-1,1)))
            else:
                for pred in preds:
                    pred = pred.reshape((1,-1))
                    formatted_preds.append(pred)
                    preds_mean.append(pred)
                    preds_std.append(np.ones(pred.shape)*np.nan)
         

            #print('preds shape after average: {}'.format(preds_mean.shape))
            return formatted_preds, preds_mean, preds_std

        def get_labels(dirs_file, out_list):
            acts = self._get_labels(dirs_file, labels_store,
                                    out=out_list)
            return acts

        for ix, dirs_file in provided_dirs:
            print('working on directory list: {}'.format(dirs_file))
            X = np.loadtxt(dirs_file, dtype=np.str).flatten()
            y_preds, y_mean, y_std = get_preds(dirs_file)
            y_act = get_labels(dirs_file, out_list)

            #print('X shape: {}'.format(X.shape))
            #print('preds length: {}'.format(len(y_preds)))
            #print('preds first elem shape: {}'.format(y_preds[0].shape))

            for j, col in enumerate(out_list):
                #print('working on col num {} named {}'.format(j,col))
                #print('yact.shape: {}'.format(y_act[j].flatten().shape))
                #print('ymean.shape: {}'.format(y_mean[j].flatten().shape))
                #print('ystd.shape: {}'.format(y_std[j].flatten().shape))

                pd_dict = {'molecule': X, 'actual': y_act[j].flatten(), 'predicted': y_mean[j].flatten(), 
                           'std': y_std[j].flatten()}
                pd_orient = {'orientation_{:02d}'.format(i):vals.flatten() for i,vals in enumerate(y_preds[j])}
                pd_dict.update(pd_orient)
            
                df = pd.DataFrame(pd_dict)
                save_file = os.path.join(save_dir, '{}.h5'.format(names[ix]))
                df.to_hdf(save_file, col)
                print('df {} result saved to {}!'.format(col, save_file))


    @staticmethod
    def _get_labels(dirs_file, labels_store, out):
        df = pd.read_hdf(labels_store, 'dfstore') 
        labels_array = []
        dirs = np.loadtxt(dirs_file, dtype=np.str)
        for dir_name in dirs:
            df_idx = int(dir_name[2:])
            labels_array.append(df[out].loc[df_idx])
       
        y = np.asarray(labels_array)
        list_of_output_vectors = [v for v in y.transpose()]
       
        return list_of_output_vectors

