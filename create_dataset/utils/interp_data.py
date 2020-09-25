import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import multiprocessing as mp
import tqdm
import os

# set up new extent
n_x = 64
n_y = 64
n_z = 64

spacing = 0.4

# set up extent grid
x = (n_x - 1)/2. * spacing
y = (n_y - 1)/2. * spacing
z = (n_z - 1)/2. * spacing

# !Note! that x and y are 'switched' to account for how arrays are filled 
#grid_iy, grid_ix, grid_iz = np.meshgrid(np.linspace(-y,y,n_y), np.linspace(-x,x,n_x), np.linspace(-z,z,n_z))
grid_ix, grid_iy, grid_iz = np.meshgrid(np.linspace(-x,x,n_x), np.linspace(-y,y,n_y), np.linspace(-z,z,n_z), 
                                        indexing='ij')
pts = np.hstack((grid_ix.reshape((-1,1)), grid_iy.reshape((-1,1)), grid_iz.reshape((-1,1))))

# create simple function to generate 1d arrays for grid
def arr(x0, step, num):
    return np.arange(0,int(num))*step+float(x0)

# create interpolation function
def interp(dirname):
    dirname = dirname[:-1] #get rid of enline character
   
    i_cube_filename = os.path.join(DATA_DIR, '{}'.format(dirname), 'interp.cube')
    i_pot_filename = os.path.join(DATA_DIR, '{}'.format(dirname), 'interp.pot')
    cube_filename = os.path.join(DATA_DIR, '{}'.format(dirname), 'input_density.cube')
    pot_filename = os.path.join(DATA_DIR, '{}'.format(dirname), 'input_density.pot')
    npy_cube = os.path.join(NPY_DIR, '{}'.format(dirname), 'cube')
    npy_pot = os.path.join(NPY_DIR, '{}'.format(dirname), 'pot')
    interp_cube = open(i_cube_filename, 'w')
    interp_pot = open(i_pot_filename, 'w')
    cube_file = open(cube_filename, 'r')
    pot_file = open(pot_filename, 'r')

    # need to make directory for molecule or the npy save will not work
    if not os.path.exists(os.path.dirname(npy_cube)):
       os.mkdir(os.path.dirname(npy_cube))

    # grab header in the first two lines
    h1 = cube_file.readline()
    h2 = cube_file.readline()
    next(pot_file)
    next(pot_file)

    # read off number atoms and starting location
    n_atoms, x0, y0, z0 = cube_file.readline().split()
    nx, x1, x2, x3 = cube_file.readline().split()
    ny, y1, y2, y3 = cube_file.readline().split()
    nz, z1, z2, z3 = cube_file.readline().split()

    xr = max(float(x1), float(x2), float(x3))
    yr = max(float(y1), float(y2), float(y3))
    zr = max(float(z1), float(z2), float(z3))

    # skip over the same data in pot file
    for _ in range(4):
        next(pot_file)

    # grab atomic data
    atomic_lines = []
    for _ in range(int(n_atoms)):
        atomic_lines.append(cube_file.readline())
        next(pot_file)
    
    # read and store values in arrays
    cube_vals = np.zeros((int(nx), int(ny), int(nz)))
    pot_vals = np.zeros((int(nx), int(ny), int(nz)))
    for i in range(int(nx)):
        for j in range(int(ny)):
            z_list_cube = [] 
            z_list_pot = []
            for _ in range(int(np.ceil(int(nz)/6.))):
                z_list_cube += cube_file.readline().split()
                z_list_pot += pot_file.readline().split()
            z_list_cube = np.array(z_list_cube).astype(np.float)
            z_list_pot = np.array(z_list_pot).astype(np.float)
            cube_vals[i,j,:] = z_list_cube
            pot_vals[i,j,:] = z_list_pot
  
    cube_file.close()
    pot_file.close()

    # call scipy interpolation function
    gx = arr(x0, xr, nx)
    gy = arr(y0, yr, ny)
    gz = arr(z0, zr, nz)

    cube_interpolator = RGI((gx,gy,gz), cube_vals, bounds_error=False, fill_value=0.0)
    i_vals_cube = cube_interpolator(pts).reshape((n_x, n_y, n_z))

    pot_interpolator = RGI((gx,gy,gz), pot_vals, bounds_error=False, fill_value=0.0)
    i_vals_pot = pot_interpolator(pts).reshape((n_x, n_y, n_z))

    # write results to file
    interp_cube.write(h1)
    interp_cube.write(h2)
    interp_cube.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(int(n_atoms),-x,-y,-z))
    interp_cube.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_x,spacing,0.,0.))
    interp_cube.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_y,0.,spacing,0.))
    interp_cube.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_z,0.,0.,spacing))

    interp_pot.write(h1)
    interp_pot.write(h2)
    interp_pot.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(int(n_atoms),-x,-y,-z))
    interp_pot.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_x,spacing,0.,0.))
    interp_pot.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_y,0.,spacing,0.))
    interp_pot.write('{:5d}{:12.6f}{:12.6f}{:12.6f}\n'.format(n_z,0.,0.,spacing))
    
    for a_line in atomic_lines:
        interp_cube.write(a_line)
        interp_pot.write(a_line)

    # calculate number of rows to print at a time with 6 to a column
    n_rows = int(np.ceil(n_z/6))
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_rows):
                interp_cube.write(''.join('{:13.5E}'.format(val) for val in i_vals_cube[i,j,k*6:(k+1)*6]))
                interp_cube.write('\n')
                interp_pot.write(''.join('{:13.5E}'.format(val) for val in i_vals_pot[i,j,k*6:(k+1)*6]))
                interp_pot.write('\n')            

    interp_cube.close()
    interp_pot.close()
    np.save(npy_cube, i_vals_cube)
    np.save(npy_pot, i_vals_pot)
    
    #print('finished {}'.format(dirname))

# get list of dirs and its length
dirname_read = open('../../ml_models/data/molecule_list.txt','r')
dirs_list = dirname_read.readlines()
n_dirs = len(dirs_list)

DATA_DIR = os.path.abspath('../data')
NPY_DIR = os.path.abspath('../../ml_models/data/molecules')

print('going to read data from {}'.format(DATA_DIR))
print('going to write .npy to {}'.format(NPY_DIR))

# apply interpolation to list of files
pool = mp.Pool(mp.cpu_count())
print('Beginning interpolation...')
print('Found {} files to interpolate.'.format(n_dirs))
print('spawning {} processes!'.format(mp.cpu_count()))
print('Note that progress bar location is accurate,')
print(" but 'time remaining' may not be due to the mutliprocessing.\n")
#chunk_size = int(max(np.floor(n_dirs/(mp.cpu_count()*5)), 1))

pool.imap_unordered(interp, tqdm.tqdm(dirs_list)) #, chunksize=chunk_size)
pool.close()
pool.join()

dirname_read.close()

