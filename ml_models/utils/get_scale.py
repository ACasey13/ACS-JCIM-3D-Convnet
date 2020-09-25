import numpy as np
import multiprocessing as mp
import os
from functools import partial

data_parent_dir = os.path.abspath('../data/molecules/')
dirs_file = os.path.abspath('../data/train_dirs.txt')

def read_cube(dir_name, cube_type):
    f_name = os.path.join(data_parent_dir, dir_name, 'npy_{}.npy'.format(cube_type))
    vals = np.load(f_name) 
    mi = np.min(vals)
    ma = np.max(vals)
    return mi, ma

def get_scale(dirs_file, cube_type):
    cube_res = []
    def cube_r(result):
        cube_res.append(result)
    
    dirs = np.loadtxt(dirs_file, dtype=np.str)
    n_cubes = len(dirs)

    func = partial(read_cube, cube_type=cube_type)

    print('found {} named directories!'.format(n_cubes))
    print('reading in {}s and storing min and max values...'.format(cube_type))
    print('spawning {} processes!'.format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    pool.map_async(func, dirs, callback=cube_r)
    pool.close()
    pool.join()
    print(len(cube_res))
    cube_res = np.array(cube_res)
    c_mi = np.min(cube_res)
    c_ma = np.max(cube_res)
    cube_scale = (c_mi, c_ma)
    return cube_scale

if __name__ == '__main__':
    res = get_scale(dirs_file, 'pot')
    print('pot: {}'.format(res))
    res =get_scale(dirs_file, 'cube')
    print('cube: {}'.format(res))
