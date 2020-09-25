import numpy as np
import pandas as pd
#from tensorflow.python import keras
import keras
import os

data_parent_dir = os.path.abspath('../data/molecules')

# create data generator
# right now, all the data preprocessing steps are done 
# in the generator; which is fine.
# The issue is that the list of arguments becomes long,
# and will only get longer, as more preprocessing steps
# or ideas are carried out.
# A better solution might be to create an sklearn 
# pipeline containing all of the preprocessing information
# (could have a pipeline for .cube and .pot files)
# and then pass the pipeline to the generator
class GenCreatorMulti(keras.utils.Sequence):
    def __init__(self,
                 batch_size,
                 dirs_file,
                 labels_store,
                 scale_cube=1.,
                 scale_pot=1.,
                 augment=True,
                 shuffle=True,
                 dshape='rect',
                 out_list = ['density', 'detvel', 'detpres',
                             'gp2', 'energy', 'heat',
                             'cjtemp', 'gp1'], 
                 parent_dir=data_parent_dir,
                 verbose=0,
                 name=None,
                 cube_clip=None,
                 pot_clip=None):
        
        self.batch_size = batch_size
        self.augment = augment
        self.dshape = dshape
        self.parent = parent_dir
        self.dirs_file = dirs_file
        self.scale_cube = scale_cube
        self.scale_pot = scale_cube
        self.labels = pd.read_hdf(labels_store, 'dfstore')
        self.shuffle = shuffle
        self.out_list = out_list
        self.verbose = verbose

        if name != None:
            self.name = name
        else:
            self.name = dirs_file

        self.cube_clip = cube_clip
        self.pot_clip = pot_clip
        
        self.dirs = np.loadtxt(self.dirs_file, dtype=np.str)

        self.aug_dirs = self._augment_dir_names(self.dirs, self.augment, self.dshape)
      
        self.n_examples = len(self.aug_dirs)
        
        # shuffle the set before the first epoch begins
        self.on_epoch_end()

        print('Instantiated data generator {}'.format(self.name))
        print('{} examples found in {}'.format(self.n_examples, self.dirs_file))

    def __len__(self):
        # Denotes the number of batches per epoch
        l = int(np.ceil(self.n_examples / self.batch_size))
        return l

    def __getitem__(self, index):
        # get batch of data
        batch_dirs = self.aug_dirs[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(batch_dirs, index)
        return X, y

    def on_epoch_end(self):
        if self.verbose:
            print('\nIn {}, Epoch ended!!\n'.format(self.name))
        # Updates indexes after each epoch
        if self.shuffle == True:
            if self.verbose:
                print('shuffling {}!\n'.format(self.name))
            np.random.shuffle(self.aug_dirs)

    def __data_generation(self, batch_dirs, index):
        # Generates data containing batch_size samples
        cube_array = []
        labels_array = []

        for act_dir in batch_dirs:
            dir_name = act_dir[:-3]
            orientation = int(act_dir[-2:])
            df_idx = int(dir_name[2:])
            labels_array.append(self.labels[self.out_list].loc[df_idx])
            cube_name = os.path.join(self.parent, '{}'.format(dir_name), 'cube.npy')
            pot_name = os.path.join(self.parent, '{}'.format(dir_name), 'pot.npy')
            orig_cube = self._read_cube(cube_name) 
            orig_pot = self._read_cube(pot_name) 

            # clip if requested
            if self.cube_clip != None:
                orig_cube[orig_cube > self.cube_clip] = self.cube_clip
            if self.pot_clip != None:
                orig_pot[orig_pot > self.pot_clip] = self.pot_clip
            
            # scale accordingly
            orig_cube *= self.scale_cube
            orig_pot *= self.scale_pot

            # rotate accordingly
            orig_cube = self._rotate_cube(orig_cube, orientation, self.dshape)
            orig_pot = self._rotate_cube(orig_pot, orientation, self.dshape)
            
            cube_array.append(np.concatenate((orig_cube[:,:,:,np.newaxis],
                                               orig_pot[:,:,:,np.newaxis]), axis=3))
                
        y = np.asarray(labels_array)
        list_of_output_vectors = [v for v in y.transpose()]

        return np.array(cube_array), list_of_output_vectors
    
    @staticmethod
    def _read_cube(f_name):
        vals = np.load(f_name)
        return vals

    @staticmethod
    def _rotate_cube(cube, orientation, cube_shape):
        if cube_shape == 'rect':
            if orientation == 1:
                cube = np.rot90(cube, 2, axes=(0,1))
            elif orientation == 2:
                cube = np.rot90(cube, 2, axes=(1,2))
            elif orientation == 3:
                cube = np.rot90(cube, 2, axes=(0,2))
        
        unique_cube_states = ((0, 0, 0), (0, 0, 1), (0, 0, 2),
                              (0, 0, 3), (0, 1, 0), (0, 1, 1),
                              (0, 1, 2), (0, 1, 3), (0, 2, 0),
                              (0, 2, 1), (0, 2, 2), (0, 2, 3),
                              (0, 3, 0), (0, 3, 1), (0, 3, 2),
                              (0, 3, 3), (1, 0, 0), (1, 0, 1),
                              (1, 0, 2), (1, 0, 3), (1, 2, 0),
                              (1, 2, 1), (1, 2, 2), (1, 2, 3))
    
        if cube_shape == 'cube':
            state = unique_cube_states[orientation]
            cube = np.rot90(cube, state[0], axes=(1,2))
            cube = np.rot90(cube, state[1], axes=(2,0))
            cube = np.rot90(cube, state[2], axes=(0,1))
                
        return cube
                

    @staticmethod
    def _augment_dir_names(dir_names, augment, cube_shape):
        # if 3D data is a cube (equal length in all dimensions)
        # then there are 24 possible unique orientations that can be 
        # formed with 90 degree rotations 
        # if the 3D dats is rectangular (unequal side lengths)
        # then there are 4 unique rotations which are formed with
        # 180 degree rotations (presevering initial shape!) 
        # augment creates 'virtual' data file references so that
        # the data can be re-oriented at read time

        aug_dirs = np.array([], dtype=np.str)
        orientations = {'rect': 4, 'cube': 24}
        if augment:
            for i in range(orientations[cube_shape]):
                aug_dirs = np.hstack((aug_dirs, np.char.add(dir_names, u'_{:02d}'.format(i))))
        else:
            aug_dirs = np.char.add(dir_names, u'_00')
        return aug_dirs




