import os
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import fcntl
import pandas as pd
import numpy as np
from collections import Counter

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


H5_FILE = '../non_volatile_data/dh-data_v2.h5'
LOCK_FILE = '../non_volatile_data/df.lock'
h5_table = 'dfstore'

lock_filename = LOCK_FILE
lock = Lock(lock_filename)

try:
    lock.acquire()
    df = pd.read_hdf(H5_FILE, h5_table)
    if 'status' not in df.columns:
        # set initial status to -1
        df.insert(0, 'status', pd.Series(np.zeros(len(df), dtype=np.int)-1, index=df.index))
    #df['Flag'] = df['Flag'].astype(np.int)
    for idx in df.index:
        #print(df.loc[idx]['Flag'])
        #print(type(df.loc[idx]['Flag']))
        if df.loc[idx]['Flag'] == 0:
            continue
        try:
            smile = df.loc[idx]['smiles']
            m = Chem.MolFromSmiles(smile)
            m = Chem.AddHs(m)
            c_smile = Chem.MolToSmiles(m)
            df.loc[idx, 'c_smiles'] = c_smile
            df.loc[idx, 'status'] = 0
        except:
            df.loc[idx, 'status'] = -2
            print("could not convert smile '{}' of molecule {} : {}".format(smile, idx, df.loc[idx]['Name']))
    f_name = H5_FILE[:-3] + '-canon.h5'
    df.to_hdf(f_name, 'dfstore', format='table')
except:
    raise Exception('lock section error')
finally:
    lock.release()

