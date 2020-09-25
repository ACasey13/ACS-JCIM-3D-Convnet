#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:41:14 2018

@author: bbarnes

Modified by acasey on Mon Jul 7 2019
"""

import os
import sys
import time
import fcntl
from collections import Counter
import subprocess
import numpy as np
import pandas as pd
from pandas import HDFStore
from rdkit import Chem
#from parse_mol_data import load_data
from mol_descriptors import calc_oxy_bal
from basic_conformers import make_conformer, _extract_atomic_type, _atomic_pos_from_conformer, write_xyz
from get_edat_prediction import calculate_edat


"""
First, a major assumption being made is that this script will be called from within 
its current directory. This is because relative paths will be used.
I think this is easier for PBS job submissions, because the cd call can be made at 
the shell (system) level before calling python pipehead.
"""

LOCK_FILE = '../non_volatile_data/df.lock'
H5_FILE = '../non_volatile_data/cands-concat-clean.h5'


def run_test_case(rewrite=1):
    # RDX

    rdx = [0, 'C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]',
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data = [rdx,]
    df = pd.DataFrame(data, columns=['status', 'smiles', 'density',
                      'heat', 'energy', 'exenergy', 'detvel', 
                      'detpres', 'cjtemp', 'gp1', 'gp2', 'gp3',
                      'dipole_x', 'dipole_y', 'dipole_z', 'hf'])

    if rewrite:
        df.to_hdf('../non_volatile_data/test_case.h5', 'dfstore', format = 'table')
    else:
        df = pd.read_hdf('../non_volatile/test_case.h5','dfstore')
        df['status'] = 2
        df.to_hdf('../non_volatile_data/test_case.h5', 'dfstore', format='table')
    process_next('../non_volatile_data/test_case.h5', test=1)


# via http://blog.vmfarms.com/2011/03/cross-process-locking-and.html
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


def process_next(h5_file, test=0):
    h5_table = 'dfstore'
    lock_filename = LOCK_FILE
    lock = Lock(lock_filename)
    header('In Gaussian Section')
    try:
        lock.acquire()
        df = pd.read_hdf(h5_file, h5_table)
        # find the next item to process
        items = df.index[df['status'] == 0]
        if len(items) > 0:
            ns_loc = items[0]
            next_smiles = df.loc[ns_loc]['smiles']
            print('working on....')
            print('index: {:3d} smile: {}'.format(ns_loc, next_smiles))
            df.loc[df['smiles'] == next_smiles, 'status'] = 1
            df.to_hdf(h5_file, h5_table, format = 'table')
        else:
            print('all smiles have finished gaussian')
    except:
        raise Exception('lock section error')
    finally:
        lock.release()

    if 'next_smiles' in locals():
        if test:
            dir_name = 'test_case'
        else:
            dir_name = 'm_{:06d}'.format(ns_loc)
        m = next_smiles
        m = Chem.MolFromSmiles(str(m))
        m = make_conformer(m)
        my_conf = m.GetConformers()[0]
        pos = _atomic_pos_from_conformer(my_conf)
        elements = _extract_atomic_type(my_conf)
        pos = [[-float(coor[k]) for k in range(3)] for coor in pos]
        coords = list(zip(elements, pos))
        edat_worked,density,heat,band_gap,dipole,mol_volume,energy,dx,dy,dz,hf=calculate_edat(next_smiles,elements,pos,dir_name)

        if(edat_worked): 
            try:
                lock.acquire()
                df = pd.read_hdf(h5_file, h5_table)
                df.loc[df['smiles'] == next_smiles, 'status'] = 2
                df.loc[df['smiles'] == next_smiles, 'density'] = density
                df.loc[df['smiles'] == next_smiles, 'heat'] = heat
                df.loc[df['smiles'] == next_smiles, 'energy'] = energy
                df.loc[df['smiles'] == next_smiles, 'gp1'] = band_gap
                df.loc[df['smiles'] == next_smiles, 'gp2'] = dipole
                df.loc[df['smiles'] == next_smiles, 'gp3'] = mol_volume
                df.loc[df['smiles'] == next_smiles, 'dipole_x'] = dx
                df.loc[df['smiles'] == next_smiles, 'dipole_y'] = dy
                df.loc[df['smiles'] == next_smiles, 'dipole_z'] = dz
                df.loc[df['smiles'] == next_smiles, 'hf'] = hf

            except:
                raise Exception('error on lock after edat success')
            finally:
                df.to_hdf(h5_file, h5_table, format = 'table')
                print('edat success, store updated')
                lock.release()
        else:
            try:
                lock.acquire()
                df = pd.read_hdf(h5_file, h5_table)
                df.loc[df['smiles'] == next_smiles, 'status'] = 3
            except:
                raise Exception('error on lock after edat failure')
            finally:
                df.to_hdf(h5_file, h5_table, format = 'table')
                print('edat failed')
                lock.release()
                sys.exit()

def header(title):
    print('')
    print('*'*50)
    print(title)
    print('*'*50)

#run_test_case(1)
process_next(H5_FILE)
