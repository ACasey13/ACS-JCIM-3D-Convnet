# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:31:59 2017

@author: brian.c.barnes2.civ
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# m is a rdkit molecule with hydrogens
# manual really recommends new ETKDG method

#m = 'Cc1cc([N+](=O)[O-])cc(C)c1C'
#m = Chem.MolFromSmiles(str(m))
def make_conformer(m):
    m = Chem.AddHs(m)

    cids = AllChem.EmbedMultipleConfs(m, 1000, AllChem.ETKDG())
    props = AllChem.MMFFGetMoleculeProperties(m)
    energies = []
    print('searching for best conformer....')
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(m, confId=cid)  
        potential = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
        mmff_energy = potential.CalcEnergy()
        energies.append((cid, mmff_energy))
    energies = np.asarray(energies)
    best_energy = np.min(energies[:,1])
    best_cid = int(energies[np.argmin(energies[:,1]),0])
    print('conformer found', best_cid, best_energy)
    for cid in cids:
        if cid != best_cid:
            m.RemoveConformer(cid)
#    for a_conf in m.GetConformers():
#        cid = a_conf.GetId()
#        AllChem.MMFFOptimizeMolecule(m, confId=cid)
#        potential = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
#        print(cid)
#        mmff_energy = potential.CalcEnergy()
#        print(mmff_energy)
    return m

#m = make_conformer(m)
#my_conf = m.GetConformers()[0]

def _extract_atomic_type(confomer):
    '''
    Extracts the elements associated with a conformer, in order that prune_threshy
    are read in
    '''
    elements = []
    mol = confomer.GetOwningMol()
    for atom in mol.GetAtoms():
        elements.append(atom.GetSymbol())
    return elements


def _atomic_pos_from_conformer(conformer):
    '''
    Extracts the atomic positions for an RDKit conformer object, to allow writing
    of input files, uploading to databases, etc.
    Returns a list of lists
    '''
    atom_positions = []
    natoms = conformer.GetNumAtoms()
    for atom_num in range(0, natoms):
        pos = conformer.GetAtomPosition(atom_num)
        atom_positions.append([pos.x, pos.y, pos.z])
    return atom_positions

    
def write_xyz(coords, filename, comment):
    '''
    Write an xyz file from coords
    '''
    with open(filename, "w") as fp:
        fp.write(str(len(coords))+"\n")
        fp.write(str(comment)+"\n")
        for atom in coords:
            fp.write("%s %.4f %.4f %.4f\n" % (atom[0], atom[1][0], atom[1][1], atom[1][2]))

