# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:40:16 2018

@author: brian.c.barnes2.civ
"""
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Descriptors, AllChem

def calc_oxy_bal(a_mol):
    '''calculates oxygen balance, a quantity often used to describe EMs'''
    molwt = Descriptors.MolWt(a_mol)
    num_c = 0
    num_h = 0
    num_n = 0
    num_o = 0
    other = 0
    for atom in a_mol.GetAtoms():
        a_type = atom.GetAtomicNum()
        if a_type == 6:
            num_c += 1
        elif a_type == 1:
            num_h += 1
        elif a_type == 7:
            num_n += 1
        elif a_type == 8:
            num_o += 1
        else:
            other += 1

    oxb = -1600.0/molwt*(2.0*num_c+num_h/2.0-num_o)
    a_types = [num_c, num_h, num_n, num_o, other]
    return oxb, a_types



def morgan_bits(samps):
    '''operates on a nested list to spit out morgan fp bit vectors'''
    mols = [s[1] for s in samps]
    fpvecs = [AllChem.GetMorganFingerprintAsBitVect(m, 1) for m in mols]
    return fpvecs

def make_bits(samps):
    '''another operator on data from find_candidates()'''
    fpvecs = [AllChem.GetMorganFingerprintAsBitVect(s, 2, nBits=1024) for s in samps]
    return fpvecs

def obs_to_np(samps):
    '''another operator on data from find_candidates()'''
    obs = [s[2] for s in samps]
    obvs = np.asarray(obs)
    return obvs

def atypes_to_np(samps):
    '''another operator on data from find_candidates()'''
    atypes = [s[3] for s in samps]
    atypes = np.asarray(atypes)
    return atypes

def mbvecs_to_np(mbvecs):
    '''used to convert fingerprint data to a numpy array for ML algos'''
    converted = []
    for fprint in mbvecs:
        arr = np.zeros(1,)
        DataStructs.ConvertToNumpyArray(fprint, arr)
        converted.append(arr)
    converted = np.asanyarray(converted)
    return converted

