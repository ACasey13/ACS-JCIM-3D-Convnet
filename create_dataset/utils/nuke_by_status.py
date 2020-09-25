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

def nuke_dirs(df, i):
    if i == 5:
        # you probably did not want to nuke all successes
        # this forces you to do it yourself
        print('really? nah')
        return
    #deathpool = list(df.loc[df['status'] == i, 'smiles'])
    deathpool = df.index[df['status'] == i]
    print('removing %d directories' % len(deathpool))
    for idx in deathpool:
        dir_name = os.path.abspath('../dh_data/m_{:06d}'.format(idx))
        if verbose:
            print('removing {}!'.format(dir_name))
        #dname2=smile_string.replace('(','Q').split()
        #dname=dname2[0].replace(')','Z').split()[0]
        try:
            shutil.rmtree(dir_name)
        except:
            print('error on {}'.format(dir_name))


def nuke_status(df, i):
    if i == 5:
        # you probably did not want to nuke all successes
        # this forces you to do it yourself
        print('really? nah')
        return df
    print('nuking %d status entries' % len(df.index[df['status'] == i]))
    df.loc[df['status'] == i, ['density']] = 0.0
    df.loc[df['status'] == i, ['heat']] = 0.0
    df.loc[df['status'] == i, ['exenergy']] = 0.0
    df.loc[df['status'] == i, ['detvel']] = 0.0
    df.loc[df['status'] == i, ['detpres']] = 0.0
    df.loc[df['status'] == i, ['cjtemp']] = 0.0
    df.loc[df['status'] == i, ['gp1']] = 0.0
    df.loc[df['status'] == i, ['gp2']] = 0.0
    df.loc[df['status'] == i, ['gp3']] = 0.0
    df.loc[df['status'] == i, ['energy']] = 0.0
    df.loc[df['status'] == i, ['dipole_x']] = 0.0
    df.loc[df['status'] == i, ['dipole_y']] = 0.0
    df.loc[df['status'] == i, ['dipole_z']] = 0.0
    df.loc[df['status'] == i, ['hf']] = 0.0
    df.loc[df['status'] == i, ['status']] = 0
    return df

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

status_to_reset = 7

verbose = True

print('Going to nuke dirs and statuses for status = {} in {}'.format(status_to_reset, H5_FILE))

try:
    lock.acquire()
    df = pd.read_hdf(H5_FILE, 'dfstore')
    nuke_dirs(df, status_to_reset)
    print('\n' + '-'*50 + '\n')
    nuke_status(df, status_to_reset)
    df.to_hdf(H5_FILE, 'dfstore', format = 'table')
except:
    raise
finally: 
    lock.release()

print('Done with nuking!')
