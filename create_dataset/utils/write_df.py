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


default_h5 = '../non_volatile_data/cands-concat-clean.h5'

parser = argparse.ArgumentParser(description='Write .h5 file to text file')
parser.add_argument("filepath", nargs='?',
                    help="filepath to the .h5 file to be written to txt",
                    default=default_h5)
parser.add_argument("-d", "--destination",
                    help="name of destination file")
args = parser.parse_args()

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

if args.destination:
    outfile = args.destination
else:
    outfile = os.path.split(args.filepath)[1].split('.')[0] + '.txt'
    outfile = os.path.join('../inspection', outfile)

try:
    lock.acquire()
    df = pd.read_hdf(args.filepath)
    df.to_csv(outfile, sep=',')
    print(f'Wrote {args.filepath} to {outfile}!') 
except Exception as e:
    print(f'Error opening file {args.filepath}. Exception: {e}')
    raise
finally: 
    lock.release()

