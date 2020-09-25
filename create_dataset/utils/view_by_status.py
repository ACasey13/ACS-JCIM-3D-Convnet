#!/usr/bin/env python3

import pandas as pd
import shutil
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

H5_FILE = '../non_volatile_data/dh-data_v2-canon-clean.h5'

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

status = [3, 6, 7]

try:
    lock.acquire()
    df = pd.read_hdf(H5_FILE, 'dfstore')
    idx = []
    for i in status:
        l = list(df.index[df['status']==i])
        print(l)
        idx += l

    df = df.loc[idx]

    condensed_fname = H5_FILE[:-3] + '-view.csv'

    print('Head of df view:')
    print(df.head())

    df.to_csv(condensed_fname)

except:
    raise
finally: 
    lock.release()

print('Done with editing!')
