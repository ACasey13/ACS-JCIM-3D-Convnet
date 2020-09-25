import numpy as np
import matplotlib.pyplot as plt
import os

good_dirs = '../../ml_models/data/molecule_list.txt'

x_res = []
y_res = []
z_res = []
x_n = []
y_n = []
z_n = []
dx = []
dy = []
dz = []
bins = 100

f_dir = open(good_dirs,'r')
failed = 0
not_found = 0
for dir_name in f_dir.readlines():
    successfull_read = 0
    cube = '../data/{}/input_density.cube'.format(dir_name[:-1])
    print('reading in cube: {}'.format(cube))
    try:
        f_c = open(cube, 'r')
     
        # move past header
        next(f_c)
        next(f_c)
  
        # get atom info
        n_atoms, x0, y0, z0 = f_c.readline().split()
        nx, x1, x2, x3 = f_c.readline().split()
        ny, y1, y2, y3 = f_c.readline().split()
        nz, z1, z2, z3 = f_c.readline().split()
        successfull_read = 1
    except FileNotFoundError:
        not_found += 1
        failed += 1
    except Exception as e:
        print('file load failed for {}'.format(os.path.abspath(cube)))
        print('error: {}'.format(e))
        failed += 1
    finally:
        f_c.close()

    if successfull_read:
        x_n.append(int(nx)); y_n.append(int(ny)); z_n.append(int(nz))

        x_r = max(float(x1), float(x2), float(x3))
        y_r = max(float(y1), float(y2), float(y3))
        z_r = max(float(z1), float(z2), float(z3))
        x_res.append(x_r); y_res.append(y_r); z_res.append(z_r)

        dx.append((int(nx) - 1) * x_r)
        dy.append((int(ny) - 1) * y_r)
        dz.append((int(nz) - 1) * z_r)

f_dir.close()

print('\nnumber of directories that failed to be opened: {}'.format(failed))
print('number of directories not found: {}\n'.format(not_found))

max_extent = max(max(dx),max(dy),max(dz))
print('\nMaximum extent is: {}'.format(max_extent))

x_n = np.array(x_n); y_n = np.array(y_n); z_n = np.array(z_n)
n_pts = np.vstack((x_n, y_n, z_n))

x_res = np.array(x_res); y_res = np.array(y_res); z_res = np.array(z_res)
res = np.vstack((x_res, y_res, z_res))

np.save(os.path.join('.', 'n_pts.npy'), n_pts)
np.save(os.path.join('.', 'resolution.npy'), res)

print('files saved to cwd!')

"""
# plot historgrams
from plot_settings import *

def plot_hist(save_path, data, xlabel=None,
                               ylabel=None,
                               label=None,
                               alpha=1):
    fig, ax = plt.subplots()
    if label is None:
        ax.hist(data, bins=bins, alpha=alpha)
    else:
        ax.hist(data[0], bins=bins, label=label[0], alpha=alpha)
        ax.hist(data[1], bins=bins, label=label[1], alpha=alpha)
        ax.hist(data[2], bins=bins, label=label[2], alpha=alpha)
        ax.legend(edgecolor='w')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt_format(ax)
    plt.savefig(save_path)

save_path = '../figs/extent.png'
data = [dx, dy, dz]
labels=['x', 'y', 'z']
xlabel='Length ($\AA$)'
ylabel='Counts'

plot_hist(save_path, data, xlabel=xlabel,
                           ylabel=ylabel,
                           label=labels,
                           alpha=.5)
"""
