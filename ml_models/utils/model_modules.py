import keras
from keras import layers
from keras import models
from keras import Input
import numpy as np
import pandas as pd

df_file = '../data/cands-concat-clean-norm_params.h5'

# it is imperative that this is the same order as in the 'out_list'
# in data_gem_2!
cols_to_read = ['density', 'detvel', 'detpres', 'gp2', 'energy', 'heat', 'cjtemp', 'gp1']

df = pd.read_hdf(df_file, 'dfstore')

means = []


for col in cols_to_read:
    std = df.loc['std', col]
    var.append(np.fabs(1./std**2))

loss_weights = var


def tensor_input(input_shape=(64,64,64,2)):
    input_tensor = Input(shape=input_shape)
    return input_tensor

def irn2_stem(input_tensor):
    x = layers.Conv3D(32, (3,3,3), padding='valid', strides=2, activation='relu')(input_tensor)
    x = layers.Conv3D(32, (3,3,3), padding='valid', activation='relu')(x)
    x = layers.Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
   
    mp_1 = layers.MaxPooling3D((3,3,3), strides=2)(x)
    x = layers.Conv3D(96, (3,3,3), padding='valid', strides=2, activation='relu')(x)
    x = layers.concatenate([mp_1, x], axis=-1)

    b_1 = layers.Conv3D(64, (1,1,1), padding='same', activation='relu')(x)
    b_1 = layers.Conv3D(96, (3,3,3), padding='valid', activation='relu')(b_1)

    b_2 = layers.Conv3D(64, (1,1,1), padding='same', activation='relu')(x)
    b_2 = layers.Conv3D(64, (7,1,1), padding='same')(b_2)
    b_2 = layers.Conv3D(64, (1,7,1), padding='same')(b_2)
    b_2 = layers.Conv3D(64, (1,1,7), padding='same', activation='relu')(b_2)
    b_2 = layers.Conv3D(96, (3,3,3), padding='valid', activation='relu')(b_2)
    cat_2 = layers.concatenate([b_1, b_2], axis=-1)

    b_3 = layers.Conv3D(192, (3,3,3), padding='valid', strides=2, activation='relu')(cat_2)
    b_4 = layers.MaxPooling3D((3,3,3), padding='valid', strides=2)(cat_2)

    out = layers.concatenate([b_3, b_4], axis=-1)

    return out

def reduced_irn2_stem(input_tensor):
    x = layers.Conv3D(32, (3,3,3), padding='valid', strides=1, activation='relu')(input_tensor)
    x = layers.Conv3D(32, (3,3,3), padding='valid', activation='relu')(x)
    x = layers.Conv3D(64, (3,3,3), padding='same', activation='relu')(x)

    mp_1 = layers.MaxPooling3D((3,3,3), strides=2)(x)
    x = layers.Conv3D(96, (3,3,3), padding='valid', strides=2, activation='relu')(x)
    out = layers.concatenate([mp_1, x], axis=-1)
   
    return out

def irn2_ia(input_tensor):
    _, r, c, d, ch = input_tensor.shape
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.Conv3D(32, (1,1,1), padding='same', activation='relu')(x)

    b_2 = layers.Conv3D(32, (1,1,1), padding='same', activation='relu')(x)
    b_2 = layers.Conv3D(32, (3,3,3), padding='same', activation='relu')(b_2)

    b_3 = layers.Conv3D(32, (1,1,1), padding='same', activation='relu')(x)
    b_3 = layers.Conv3D(48, (3,3,3), padding='same', activation='relu')(b_3)
    b_3 = layers.Conv3D(64, (3,3,3), padding='same', activation='relu')(b_3)

    cat = layers.concatenate([b_1, b_2, b_3], axis=-1)
    cat = layers.Conv3D(ch.value, (1,1,1), padding='same', activation='linear')(cat)

    cat = layers.add([x, cat])

    return cat

def irn2_ra(input_tensor):
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.MaxPooling3D((3,3,3), padding='valid', strides=2)(x)

    b_2 = layers.Conv3D(192, (3,3,3), padding='valid', strides=2, activation='relu')(x)

    b_3 = layers.Conv3D(128, (1,1,1), padding='same', activation='relu')(x)
    b_3 = layers.Conv3D(128, (3,3,3), padding='same', activation='relu')(b_3)
    b_3 = layers.Conv3D(192, (3,3,3), padding='valid', strides=2, activation='relu')(b_3)

    cat = layers.concatenate([b_1, b_2, b_3])

    return cat


def irn2_ib(input_tensor):
    _, r, c, d, ch = input_tensor.shape
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.Conv3D(192, (1,1,1), padding='same', activation='relu')(x)

    b_2 = layers.Conv3D(128, (1,1,1), padding='same', activation='relu')(x)
    b_2 = layers.Conv3D(128, (1,1,7), padding='same', activation='linear')(b_2)
    b_2 = layers.Conv3D(128, (1,7,1), padding='same', activation='linear')(b_2)
    b_2 = layers.Conv3D(128, (7,1,1), padding='same', activation='relu')(b_2)

    cat = layers.concatenate([b_1, b_2], axis=-1)
    cat = layers.Conv3D(ch.value, (1,1,1), padding='same', activation='linear')(cat)

    cat = layers.add([x, cat])

    return cat

def irn2_rb(input_tensor):
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.MaxPooling3D((3,3,3), padding='valid', strides=2)(x)

    b_2 = layers.Conv3D(256, (1,1,1), padding='same', strides=1, activation='relu')(x)
    b_2 = layers.Conv3D(384, (3,3,3), padding='valid', strides=2, activation='relu')(b_2)

    b_3 = layers.Conv3D(256, (1,1,1), padding='same', activation='relu')(x)
    b_3 = layers.Conv3D(288, (3,3,3), padding='valid', strides=2, activation='relu')(b_3)

    b_4 = layers.Conv3D(256, (1,1,1), padding='same', activation='relu')(x)
    b_4 = layers.Conv3D(288, (3,3,3), padding='same', activation='relu')(b_4)
    b_4 = layers.Conv3D(320, (3,3,3), padding='valid', strides=2, activation='relu')(b_4)

    cat = layers.concatenate([b_1, b_2, b_3, b_4])

    return cat

def irn2_ic(input_tensor):
    _, r, c, d, ch = input_tensor.shape
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.Conv3D(192, (1,1,1), padding='same', activation='relu')(x)

    b_2 = layers.Conv3D(192, (1,1,1), padding='same', activation='relu')(x)
    b_2 = layers.Conv3D(224, (1,1,3), padding='same', activation='linear')(b_2)
    b_2 = layers.Conv3D(224, (1,3,1), padding='same', activation='linear')(b_2)
    b_2 = layers.Conv3D(224, (3,1,1), padding='same', activation='relu')(b_2)

    cat = layers.concatenate([b_1, b_2], axis=-1)
    cat = layers.Conv3D(ch.value, (1,1,1), padding='same', activation='linear')(cat)

    cat = layers.add([x, cat])

    return cat

def neck(input_tensor):
    x = layers.Activation('relu')(input_tensor)
    x = layers.AveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
   
    return x

def head(layer, name):
    x = layers.Dense(128, activation='relu')(layer)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='linear', name=name)(x)

    return x

def small_head(layer, name):
    x = layers.Dense(64, activation='relu')(layer)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1, activation='linear', name=name)(x)

    return x

def small_head_elu(layer, name):
    x = layers.Dense(64, activation='elu')(layer)
    x = layers.Dense(32, activation='elu')(x)
    x = layers.Dense(16, activation='elu')(x)
    x = layers.Dense(1, activation='linear', name=name)(x)

    return x

def naive_inception_block(layer, f1, f3, f3_out, f5, f5_out, mp_out):
    c1 = layers.Conv3D(f1, (1,1,1), padding='same', activation='relu')(layer)

    c3 = layers.Conv3D(f3, (1,1,1), padding='same', activation='relu')(layer)
    c3 = layers.Conv3D(f3_out, (3,3,3), padding='same', activation='relu')(c3)

    c5 = layers.Conv3D(f5, (1,1,1), padding='same', activation='relu')(layer)
    c5 = layers.Conv3D(f5_out, (5,5,5), padding='same', activation='relu')(c5)

    pool = layers.MaxPooling3D((3,3,3), strides=1, padding='same')(layer)
    pool = layers.Conv3D(mp_out, (1,1,1), padding='same')(pool)

    cat = layers.concatenate([c1, c3, c5, pool], axis=-1)

    return cat

def basic_reduction_block(input_tensor, b2, b2_out, b3, b3_mid, b3_out):
    x = layers.Activation('relu')(input_tensor)

    b_1 = layers.MaxPooling3D((3,3,3), padding='valid', strides=2)(x)

    b_2 = layers.Conv3D(b2, (1,1,1), padding='same', strides=1, activation='relu')(x)
    b_2 = layers.Conv3D(b2_out, (3,3,3), padding='valid', strides=2, activation='relu')(b_2)

    b_3 = layers.Conv3D(b3, (1,1,1), padding='same', activation='relu')(x)
    b_3 = layers.Conv3D(b3_mid, (3,3,3), padding='same', activation='relu')(b_3)
    b_3 = layers.Conv3D(b3_out, (3,3,3), padding='valid', strides=2, activation='relu')(b_3)

    cat = layers.concatenate([b_1, b_2, b_3])

    return cat


def naive_inception(input_shape=(64,64,64,2)):
    input_tensor = tensor_input(input_shape)
    i1 = naive_inception_block(input_tensor, 16,
                                              8, 16,
                                              8, 16,
                                             16)
    r1 = basic_reduction_block(i1, 16, 32,
                                   16, 24, 32)

    x = layers.Conv3D(128, (3,3,3), activation='relu')(r1)
    x = layers.MaxPooling3D((2,2,2))(x)


    i2 = naive_inception_block(x            , 32,
                                              16, 32,
                                              16, 32,
                                             32)
    r2 = basic_reduction_block(i2, 32, 64,
                                   32, 50, 64)

    n = layers.Conv3D(512, (3,3,3), activation='relu')(r2)
    n = layers.MaxPooling3D((2,2,2))(n)
    n = layers.Conv3D(512, (2,2,2), activation='relu')(n)
    n = layers.Flatten()(n)
    n = layers.Dropout(0.2)(n)

    n = layers.Dense(1024, activation='relu')(n)
    n = layers.Dense(512, activation='relu')(n)
    n = layers.Dense(128, activation='relu')(n)
    n = layers.Dense(64, activation='relu')(n)

    density = head(n, 'density')
    detvel = head(n, 'detvel')
    detpres = head(n, 'detpres')
    dipole = head(n, 'dipole')
    energy = head(n, 'energy')
    hof = head(n, 'hof')
    temp = head(n, 'temp')
    gap = head(n, 'gap')

    out_list = [density, detvel, detpres,
                dipole, energy, hof,
                temp, gap]

    model = models.Model(input_tensor, out_list)
    model.compile(optimizer='adam',
                  loss=['mse']*len(out_list),
                  loss_weights=loss_weights,
                  metrics=['mae'])

    return model


def vanilla_base(input_tensor):
    x = layers.Conv3D(32, (3,3,3), activation='relu')(input_tensor)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(64, (3,3,3), activation='relu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(128, (3,3,3), activation='relu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(256, (3,3,3), activation='relu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Flatten()(x)
    
    return x

def vanilla_base_elu(input_tensor):
    x = layers.Conv3D(32, (3,3,3), activation='elu')(input_tensor)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(64, (3,3,3), activation='elu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(128, (3,3,3), activation='elu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Conv3D(256, (3,3,3), activation='elu')(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Flatten()(x)

    return x

def vanilla_multi(input_shape=(64,64,64,2)):
    input_tensor = tensor_input(input_shape)
    n = vanilla_base(input_tensor)

    density = head(n, 'density')
    detvel = head(n, 'detvel')
    detpres = head(n, 'detpres')
    dipole = head(n, 'dipole')
    energy = head(n, 'energy')
    hof = head(n, 'hof')
    temp = head(n, 'temp')
    gap = head(n, 'gap')

    out_list = [density, detvel, detpres,
                dipole, energy, hof,
                temp, gap]

    model = models.Model(input_tensor, out_list)
    model.compile(optimizer='adam',
                  loss=['mse']*len(out_list),
                  loss_weights=loss_weights,
                  metrics=['mae'])
    
    return model

def vanilla_multi_2(input_shape=(64,64,64,2)):
    input_tensor = tensor_input(input_shape)
    n = vanilla_base(input_tensor)
    n = layers.Dense(256, activation='relu')(n)
    n = layers.Dense(128, activation='relu')(n)
    n = layers.Dense(64, activation='relu')(n)



    density = small_head(n, 'density')
    detvel = small_head(n, 'detvel')
    detpres = small_head(n, 'detpres')
    dipole = small_head(n, 'dipole')
    energy = small_head(n, 'energy')
    hof = small_head(n, 'hof')
    temp = small_head(n, 'temp')
    gap = small_head(n, 'gap')

    out_list = [density, detvel, detpres,
                dipole, energy, hof,
                temp, gap]

    model = models.Model(input_tensor, out_list)
    model.compile(optimizer='adam',
                  loss=['mse']*len(out_list),
                  loss_weights=loss_weights,
                  metrics=['mae'])

    return model

def vanilla_multi_2_elu(input_shape=(64,64,64,2)):
    input_tensor = tensor_input(input_shape)
    n = vanilla_base_elu(input_tensor)
    n = layers.Dense(256, activation='elu')(n)
    n = layers.Dense(128, activation='elu')(n)
    n = layers.Dense(64, activation='elu')(n)



    density = small_head_elu(n, 'density')
    detvel = small_head_elu(n, 'detvel')
    detpres = small_head_elu(n, 'detpres')
    dipole = small_head_elu(n, 'dipole')
    energy = small_head_elu(n, 'energy')
    hof = small_head_elu(n, 'hof')
    temp = small_head_elu(n, 'temp')
    gap = small_head_elu(n, 'gap')

    out_list = [density, detvel, detpres,
                dipole, energy, hof,
                temp, gap]

    model = models.Model(input_tensor, out_list)
    model.compile(optimizer='adam',
                  loss=['mse']*len(out_list),
                  loss_weights=loss_weights,
                  metrics=['mae'])

    return model



def threeDIncResNet_multi_output_model(input_shape=(64,64,64,2)):
    input_tensor = tensor_input(input_shape)
    stem = reduced_irn2_stem(input_tensor)
    ia = irn2_ia(stem)
    ra = irn2_ra(ia)
    ib = irn2_ib(ra)
    rb = irn2_rb(ib)
    ic = irn2_ic(rb)
    n  = neck(ic)
    
    density = head(n, 'density')
    detvel = head(n, 'detvel')
    detpres = head(n, 'detpres')
    dipole = head(n, 'dipole')
    energy = head(n, 'energy')
    hof = head(n, 'hof')
    temp = head(n, 'temp')
    gap = head(n, 'gap')

    out_list = [density, detvel, detpres,
                dipole, energy, hof,
                temp, gap]
    
    model = models.Model(input_tensor, out_list)
    model.compile(optimizer='adam',
                  loss=['mse']*len(out_list),
                  loss_weights=loss_weights,
                  metrics=['mae'])

    return model


if  __name__ == '__main__':
    #model = threeDIncResNet_multi_output_model()
    #model.summary()

    #model1 = vanilla_multi()
    #model1.summary()

    model2 = naive_inception()
    model2.summary()





