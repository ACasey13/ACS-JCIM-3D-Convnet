#!/usr/bin/env python3

import pandas as pd
import fcntl
import numpy as np
import argparse
import os

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

files_to_stack = ['../non_volatile_data/store-cands-cno-1-11.h5',
                  '../non_volatile_data/store-cands-cno-12-shuffled.h5',
                  '../non_volatile_data/store-cands-cno-17-shuffled.h5']

try:
    lock.acquire()
    print('reading in data frames')
    frames = [pd.read_hdf(x) for x in files_to_stack]
    print('working to stack data frames...')
    df = pd.concat(frames, ignore_index=True)
    print('saving new concatenated data frame')
    df.to_hdf('../non_volatile_data/cands-concat.h5', 'dfstore', format='table')
    print('Success!')
except Exception as e:
    raise
finally: 
    lock.release()

