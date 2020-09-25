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

parser = argparse.ArgumentParser(description='Get summary statistics on .h5 file.')
parser.add_argument("filepath", nargs='?',
                    help="filepath to the .h5 file to get summary stats on",
                    default=default_h5)
parser.add_argument("-p", "--print", action='store_true',
                    help="print summary stats to screen")
parser.add_argument("-d", "--destination",
                    help="name of destination file")
args = parser.parse_args()

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

if args.destination:
    outfile = args.destination
else:
    outfile = os.path.split(args.filepath)[1].split('.')[0] + '.summary'
    outfile = os.path.join('../inspection', outfile)


try:
        lock.acquire()
        f = open(outfile, 'w')
        df = pd.read_hdf(args.filepath)

        n_molecules = len(df)
        g_in_progress = sum(df['status']==1)
        g_success = sum(df['status']==2)
        g_failed = sum(df['status']==3)
        g_remaining = sum(df['status']==0)

        bad_smiles = sum(df['status']==7)

        c_in_progress = sum(df['status']==4)
        c_success = sum(df['status']==5)
        c_failed = sum(df['status']==6)
        c_total = n_molecules - g_failed - bad_smiles

        g_success_act = g_success + c_in_progress + c_success + c_failed
        c_remaining = c_total - (c_success + c_in_progress + c_failed)

        f.write('Total\n')
        f.write('*'*50 + '\n')
        f.write('number of molecules: {}'.format(n_molecules))
        f.write('number of bad smiles: {}\n'.format(bad_smiles))
        f.write('\nGaussian\n')
        f.write('*'*50 + '\n')
        f.write('number successfully processed: {}\n'.format(g_success_act))
        f.write('number failed attempts: {}\n'.format(g_failed))
        f.write('number in progress: {}\n'.format(g_in_progress))
        f.write('number remaining: {}\n'.format(g_remaining)) 

        f.write('\nCheetah\n')
        f.write('*'*50 + '\n')
        f.write('number successfully processed: {}\n'.format(c_success))
        f.write('number failed attempts: {}\n'.format(c_failed))
        f.write('number in progress: {}\n'.format(c_in_progress))
        f.write('number remaining: {}\n'.format(c_remaining))

        f.close()

        if args.print:
            print('Total\n')
            print('*'*50 + '\n')
            print('number of molecules: {}'.format(n_molecules))
            print('number of bad smiles: {}\n'.format(bad_smiles))

            print('\nGaussian\n')
            print('*'*50 + '\n')
            print('number successfully processed: {}\n'.format(g_success_act))
            print('number failed attempts: {}\n'.format(g_failed))
            print('number in progress: {}\n'.format(g_in_progress))
            print('number remaining: {}\n'.format(g_remaining))

            print('\nCheetah\n')
            print('*'*50 + '\n')
            print('number successfully processed: {}\n'.format(c_success))
            print('number failed attempts: {}\n'.format(c_failed))
            print('number in progress: {}\n'.format(c_in_progress))
            print('number remaining: {}\n'.format(c_remaining))

except:
        raise Exception('lock section error')
finally:
        lock.release()

