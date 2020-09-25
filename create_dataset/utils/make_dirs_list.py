import pandas as pd
import numpy as np
import os
import fcntl

class Lock:
    def __init__(self, filename):
        self.filename = filename
        # this will create it if it does not exist already
        self.handle = open(filename, 'w')

    # bitwise OR fcntl.LOCK_NB if you need a non-blocking lock
    def acquire(self):
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def release(self):
        fcntl.flock(self.handle, fcntl.LOCK_UN)

    def __del__(self):
        self.handle.close()

H5_FILE = 'cands-concat-clean.h5'

dataframe = os.path.join('../non_volatile_data', H5_FILE)
good_dirs = '../../ml_models/data/molecule_list.txt'

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

# read in dataframe
try:
    lock.acquire()
    df = pd.read_hdf(dataframe, 'dfstore')
except Exception as e:
    print('error in lock and read section')
    raise
finally:
    lock.release()

# see how many smiles are repeated
repeats = (df['smiles'].value_counts()-1).sum()
print('number of repeated smiles: {}'.format(repeats))

# get indices of completed directories
idxs = df.index[df['status'] == 5]

n_molecules = len(df)
failed = n_molecules - len(idxs)
print('number of failed molecules: {}'.format(failed))


print('saving list of good molecule directories to:\n{}...'.format(os.path.abspath(good_dirs)))
f = open(good_dirs, 'w')
for idx in idxs:
    cube = '../data/m_{:06d}/input_density.cube'.format(idx)
    pot = '../data/m_{:06d}/input_density.pot'.format(idx)
    success = 0
    if os.path.exists(cube) and os.path.exists(pot):
        success = 1
        f.write('m_{:06d}\n'.format(idx))

f.close()
print('Successful write!')
