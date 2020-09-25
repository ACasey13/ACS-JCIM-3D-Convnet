import numpy as np

train_percentage = 0.8

orig_dirs = '../data/molecule_list.txt'

dirs = np.loadtxt(orig_dirs, dtype=np.str)

n_train = int(np.floor(train_percentage * len(dirs)))
n_test = int(len(dirs) - n_train)
print('number of training examples: {}'.format(n_train))
print('number of test example: {}'.format(n_test))

np.random.shuffle(dirs)

train_dirs = dirs[:n_train]
test_dirs = dirs[n_train:]

np.savetxt('../data/train_dirs.txt', train_dirs, '%s')
np.savetxt('../data/test_dirs.txt', test_dirs, '%s')

