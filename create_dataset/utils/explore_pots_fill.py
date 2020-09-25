import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# read in files
dirname_read = open('../pipehead/perfect_dirs.txt','r')

def pot_res(result):
    global results
    results.append(result)

def read_pot(dirname):
    #print(dirname[:-1])
    dirname = dirname[:-1] #get rid of enline character
    pot_file = open('../pipehead/{}/input_density.pot'.format(dirname),'r')

    # skip first two lines
    next(pot_file)
    next(pot_file)

    # read off number atoms and starting location
    n_atoms, x0, y0, z0 = pot_file.readline().split()
    nx, x1, x2, x3 = pot_file.readline().split()
    ny, y1, y2, y3 = pot_file.readline().split()
    nz, z1, z2, z3 = pot_file.readline().split()

    xr = max(float(x1), float(x2), float(x3))
    yr = max(float(y1), float(y2), float(y3))
    zr = max(float(z1), float(z2), float(z3))

    # skip over atom data (for now)
    for _ in range(int(n_atoms)):
        next(pot_file)
    
    # read and store values in arrays
    pot_vals = np.zeros((int(nx), int(ny), int(nz)))
    for i in range(int(nx)):
        for j in range(int(ny)):
            z_list_pot = []
            for _ in range(int(np.ceil(int(nz)/6.))):
                z_list_pot += pot_file.readline().split()
            z_list_pot = np.array(z_list_pot).astype(np.float)
        
            pot_vals[i,j,:] = z_list_pot
    p_ma = np.max(pot_vals)
    p_mi = np.min(pot_vals)
    tlf = pot_vals[0,0,0]
    trf = pot_vals[0,-1,0]
    brf = pot_vals[-1,-1,0]
    blb = pot_vals[-1,0,-1]
    pot_file.close()

    return p_mi, p_ma, tlf, trf, brf, blb

if __name__ == '__main__':
    results=[]
    print('spawing {} processes!'.format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    pool.map_async(read_pot, dirname_read.readlines(), callback=pot_res)
    pool.close()
    pool.join()
    results = np.array(results)
    print('results shape: {}'.format(results.shape))
   
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(8,12))
    ax1.hist(results[0,:,0], bins=30)
    ax2.hist(results[0,:,1], bins=30)
    ax3.hist(results[0,:,2], bins=30)
    ax4.hist(results[0,:,3], bins=30)
    ax5.hist(results[0,:,4], bins=30)
    ax6.hist(results[0,:,5], bins=30)


    ax1.set_title('min')
    ax2.set_title('max')
    ax3.set_title('tlf')
    ax4.set_title('trf')
    ax5.set_title('brf')
    ax6.set_title('blb')

    plt.savefig('./pot_corner.png')   
