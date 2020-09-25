#!/usr/bin/env python3

import argparse
import pandas as pd
import fcntl
import numpy as np

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


lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

parser = argparse.ArgumentParser(description='remove duplicate molecules from a given .h5 file')
parser.add_argument("filepath", help="name of .h5 file to fix")
parser.add_argument("destination", help="name of the destination file")
args = parser.parse_args()

try:
    lock.acquire()
    df = pd.read_hdf(args.filepath)
    if 'status' not in df.columns:
        df.insert(0, 'status', pd.Series(np.zeros(len(df), dtype=np.int), index=df.index))
    repeats = (df['smiles'].value_counts()-1).sum()
    #print('number of repeated smiles: {}'.format(repeats))
    df = df.drop_duplicates(subset=['smiles'])
    df.to_hdf(args.destination, 'dfstore', format='table')
 
    print(f'Removed {repeats} repeat SMILES from {args.filepath} and wrote clean set to {args.destination}!')
except Exception as e:
    print(f'Error opening file {args.filepath}. Exception: {e}')
    raise
finally:
    lock.release()
