#!/usr/bin/env python3

import pandas as pd
import shutil
import os
import fcntl
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

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

# you must enter the edits manually in the function before running
def edit(df, idx):
    print('editing index {}...'.format(idx))
    #df.loc[idx, ['density']] = 0.0
    #df.loc[idx, ['heat']] = 0.0
    #df.loc[idx, ['exenergy']] = 0.0
    #df.loc[idx, ['detvel']] = 0.0
    #df.loc[idx, ['detpres']] = 0.0
    #df.loc[idx, ['cjtemp']] = 0.0
    #df.loc[idx, ['gp1']] = 0.0
    #df.loc[idx, ['gp2']] = 0.0
    #df.loc[idx, ['gp3']] = 0.0
    #df.loc[idx, ['energy']] = 0.0
    #df.loc[idx, ['dipole_x']] = 0.0
    #df.loc[idx, ['dipole_y']] = 0.0
    #df.loc[idx, ['dipole_z']] = 0.0
    #df.loc[idx, ['hf']] = 0.0

    #df.loc[idx, ['status']] = -1
    #df.loc[idx, 'Flag'] = 0.0
    df.loc[idx, 'h50'] = 77.5
    #df.loc[idx, ['smiles']] = "NC1=C([N+]([O-])=O)C([N+]([O-])=O)=C([N+]([O-])=O)C(C1=2)=NON2=O"
   
    #df.loc[idx, ['Formula']] = "C6H2N6O8"
    
    return df

def canon(df, idx):
    print('trying to canonize smile for idx: {}'.format(idx))
    try:
        smile = df.loc[idx]['smiles']
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)
        c_smile = Chem.MolToSmiles(m)
        df.loc[idx, 'c_smiles'] = c_smile
        df.loc[idx, 'status'] = 0

        formula = CalcMolFormula(m)
        if 'Cl' in formula:
            formula = formula.replace('Cl', '')
            formula = formula + 'Cl'

        df.loc[idx, 'Formula'] = formula

    except Exception as e:
        df.loc[idx, 'status'] = -2
        print("could not convert smile {} of molecule {} : {}".format(smile, idx, df.loc[idx]['Name']))
        print('Exception: {}'.format(e))
    return df

lock_filename = '../non_volatile_data/df.lock'
lock = Lock(lock_filename)

dry_run = True
idx = 46

print('Going to edit index {} in {}'.format(idx, H5_FILE))

try:
    lock.acquire()
    df = pd.read_hdf(H5_FILE, 'dfstore')
    old_df = df.copy()
    new = edit(df, idx)
    #new = canon(new, idx)

    k1 = set(old_df.columns)
    k2 = set(new.columns)
    k = k1.union(k2)
    print('column, old entry (left), new entry (right)')
    for col in k:
        try:
            old = old_df.loc[idx, col]
        except:
            old = 'DNE' # does not exist
        try: 
            n = new.loc[idx, col]
        except:
            n = 'DNE'
        print(f'{col}: {old}, {n}')

    if dry_run:
        print('This was a dry run. No changes made!')
    else:
        print('Changes were written to the file!')
        new.to_hdf(H5_FILE, 'dfstore', format = 'table')

except:
    raise
finally: 
    lock.release()

print('Done with editing!')
