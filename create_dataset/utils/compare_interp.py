import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

dirname = '../dh_data/m_000003/'
orig_fname = os.path.join(dirname,'input_density.cube')
interp_fname = os.path.join(dirname,'interp.cube')
# are you comparing pots or cubes?
c_type = 'cube'

# original grid array creation function
def arr(x0, step, num):
    return np.arange(0,int(num))*step+float(x0)

# read in files
interp_file = open(interp_fname, 'r')
orig_file = open(orig_fname, 'r')
  
# skip first two header lines
next(orig_file)
next(orig_file)
next(interp_file)
next(interp_file)

# read off new grid size, spacing and extent
# note the names between orig and interp are very similar
# eg; nx and xn.... (not the best practice but it works)
n_atoms_i, x, y, z = interp_file.readline().split()
xn, x1i, x2i, x3i = interp_file.readline().split()
yn, y1i, y2i, y3i = interp_file.readline().split()
zn, z1i, z2i, z3i = interp_file.readline().split()
xn = int(xn); yn = int(yn); zn = int(zn);
x = -1*float(x); y = -1*float(y); z = -1*float(z);
spacing = float(x1i)
print('interpolated cube shape: ({}, {}, {})'.format(xn, yn, zn))

   
# read off number atoms and starting location
n_atoms, x0, y0, z0 = orig_file.readline().split()
nx, x1, x2, x3 = orig_file.readline().split()
ny, y1, y2, y3 = orig_file.readline().split()
nz, z1, z2, z3 = orig_file.readline().split()
print('original cube shape: ({}, {}, {})'.format(int(nx), int(ny), int(nz)))

xr = max(float(x1), float(x2), float(x3))
yr = max(float(y1), float(y2), float(y3))
zr = max(float(z1), float(z2), float(z3))

#check if cubes have the same number of atoms...
if int(n_atoms_i) != int(n_atoms):
    raise ValueError('number of atoms in original file and interpolatd file are not the same!')

# skip over atom data (for now)
for _ in range(int(n_atoms)):
    next(orig_file)
    next(interp_file)
   
# read and store values in array 
# original data
print('reading in original values....')
orig_vals = np.zeros((int(nx), int(ny), int(nz)))
for i in range(int(nx)):
    for j in range(int(ny)):
        orig_list = [] 
        for _ in range(int(np.ceil(int(nz)/6.))):
            orig_list += orig_file.readline().split()
        orig_list = np.array(orig_list).astype(np.float)
        orig_vals[i,j,:] = orig_list
  
orig_file.close()

# interpolated data
print('reading in interpolated data....')
interp_vals = np.zeros((xn, yn, zn))
for i in range(xn):
    for j in range(yn):
        interp_list = []
        for _ in range(int(np.ceil(zn/6.))):
            interp_list += interp_file.readline().split()
        interp_list = np.array(interp_list).astype(np.float)
        interp_vals[i,j,:] = interp_list

interp_file.close()

# create original and interpolated grids
gx = arr(x0, xr, nx)
gy = arr(y0, yr, ny)
gz = arr(z0, zr, nz)
o_g_x, o_g_y = np.meshgrid(gx, gy)
i_g_x, i_g_y = np.meshgrid(np.linspace(-x,x,xn), np.linspace(-y,y,yn))
gzi = np.linspace(-z,z,zn)

# create image directory
plot_dir_name = 'compare_{}'.format(c_type)
if not os.path.exists(os.path.join(dirname, plot_dir_name)):
    os.mkdir(os.path.join(dirname, plot_dir_name))
img_path = os.path.join(dirname, plot_dir_name)

# cycle through plots
print('creating plots in directory: {}'.format(img_path))
n_levels = 100

def make_c1(ix, idx, ax1):
    vmin_o = np.min(orig_vals[:,:,idx])
    vmax_o = np.max(orig_vals[:,:,idx])
    vmax_i = np.max(interp_vals[:,:,ix])
    vmin_i = np.min(interp_vals[:,:,ix])
    vmax = max(vmax_o, vmax_i)
    vmin = min(vmin_o, vmin_i)
    levels = np.linspace(vmin, vmax, n_levels)

    c1 = ax1.contourf(o_g_x, o_g_y, np.transpose(orig_vals[:,:,closest_idx]),
                      levels, vmin=vmin, vmax=vmax)

    return c1, vmin, vmax, levels

for ix, z_val in enumerate(gzi):
    print('working on plot {}/{}'.format(ix+1, zn))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4), constrained_layout=True)
    #cb_ax = fig.add_axes([.9, 0.05, 0.025, .9])
    c1_exists = False
    closest_idx = np.argmin(np.fabs(gz-z_val))
    closest_z = gz[closest_idx]
#    if z_val < np.min(gz):
#        if np.fabs(closest_z - z_val) < spacing:
#            c1, vmin, vmax, levels = make_c1(ix, closest_idx, ax1)
#            c1_exists = True
#    elif z_val > np.max(gz):
#        if np.fabs(closest_z - z_val) < spacing:
#            c1, vmin, vmax, levels = make_c1(ix, closest_idx, ax1)
#            c1_exists = True
    if z_val >= np.min(gz) and z_val <= np.max(gz):
        c1, vmin, vmax, levels = make_c1(ix, closest_idx, ax1)
        c1_exists = True

    ax1.set_xlim(-x,x)
    ax1.set_ylim(-y,y)

    if c1_exists:
        c2 = ax2.contourf(i_g_x, i_g_y, np.transpose(interp_vals[:,:,ix]),
                          levels, vmin=vmin, vmax=vmax)
        #cb1 = fig.colorbar(c1, ax=ax1)
        cb2 = fig.colorbar(c2, ax=[ax1,ax2])

    else:
        if np.all(interp_vals[:,:,ix]==0.):
            c2 = ax2.contourf(i_g_x, i_g_y, np.transpose(interp_vals[:,:,ix]),
                          n_levels, vmin=0., vmax=10)
        else:
            c2 = ax2.contourf(i_g_x, i_g_y, np.transpose(interp_vals[:,:,ix]),
                          n_levels)
        #fig.colorbar(c2, ax=ax1)
        cb2 = fig.colorbar(c2, ax=[ax1,ax2])
    
    tick_labels = cb2.get_ticks()
    cb2.ax.set_yticklabels(['{:.3e}'.format(label) for label in tick_labels])

    ax1.set_title('Original')
    ax2.set_title('Interpolated')
    ax2.axes.get_yaxis().set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.savefig(os.path.join(img_path,'img_{:02d}.png'.format(ix)))
    plt.close(fig)
    
print('successful exit!')



   
