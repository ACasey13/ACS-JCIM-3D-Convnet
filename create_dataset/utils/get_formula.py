import os
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import fcntl
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.cm as cm

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

good_dirs = '../../ml_models/data/molecule_list.txt'
formulas = []
atom_counts = []

H5_FILE = '../non_volatile_data/cands-concat-clean.h5'
LOCK_FILE = '../non_volatile_data/df.lock'
h5_table = 'dfstore'

lock_filename = LOCK_FILE
lock = Lock(lock_filename)

try:
    lock.acquire()
    df = pd.read_hdf(H5_FILE, h5_table)
    idx = np.loadtxt(good_dirs, dtype=np.str)
    iidx = [int(i[2:]) for  i in idx]
    smiles = df.loc[iidx]['smiles']
except:
    raise Exception('lock section error')
finally:
    lock.release()

n=0
for i, smile in zip(idx, smiles):
    n += 1
    # can prepend MolWt with Exact
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    formula = CalcMolFormula(m)
    c = Counter(atom.GetSymbol() for atom in m.GetAtoms())
    atom_count = [c.get('C', 0), c.get('H', 0), c.get('N', 0), c.get('O', 0)]
    atom_counts.append(atom_count)

    if n < 11:
        print('molecule: {}  smile: {} formula: {} atom counts: {}'.format(i,smile,formula,atom_count))
    formulas.append(formula)


atom_counts = np.asarray(atom_counts)
df = pd.DataFrame(atom_counts, columns=['C', 'H', 'N', 'O'])
print('processed {} molecules!'.format(len(smiles)))
# plot historgrams
from plot_settings import *
"""
def plot_hist(save_path, data, xlabel=None,
                               ylabel=None,
                               label=None,
                               alpha=1,
                               bins=100):
    fig, ax = plt.subplots()
    if label is None:
        ax.hist(data, bins=bins, alpha=alpha)
    else:
        ax.hist(data, bins=bins, label=label, alpha=alpha)
        ax.legend(edgecolor='w')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(110,270)
    ax.set_ylim(0, 5200)
    plt_format(ax)
    plt.savefig(save_path)

save_path = '../figs/mw.png'
xlabel='Molecular Weight (g/mol)'
ylabel='Counts'

plot_hist(save_path, mws, xlabel=xlabel,
                           ylabel=ylabel,
                           alpha=.7,
                           bins=50)

"""

save_path = '../figs/atom_counts.png'

from plot_settings import *
p = sns.pairplot(df, plot_kws={'alpha':.002,'edgecolor':'k' })
#p.map_offdiag(sns.kdeplot, n_levels=100, cmap='Blues', shade=True, kernel='gau', shade_lowest=False)
p.savefig(save_path)
