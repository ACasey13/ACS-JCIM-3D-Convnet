from model_predictions_2 import ModelEval
import os
import glob

data_dir = '../data'
train_dirs = os.path.join(data_dir, 'train_dirs.txt')
test_dirs = os.path.join(data_dir, 'test_dirs.txt')
labels_store = os.path.join(data_dir, 'cands-concat-clean.h5')

models_base = os.path.abspath('./')
model_dirs = ['vanilla_multi_2_cont_3']
model_numbers = [9,]

# get latest model for each network
model_names = []
for ix, model in enumerate(model_dirs):
    saved_list = glob.glob(os.path.join(model,'model.*'))
    network_name = os.path.split(model)[1]
    num = 0
    full_model_name = ''
    if model_numbers[ix] != None:
        num = model_numbers[ix]
        full_model_name = os.path.join(model, 'model.{:02d}.hdf5'.format(model_numbers[ix]))
    else:
        for i in saved_list:
            directory, model_name = os.path.split(i)
            m_num = int(model_name.split('.')[1])
            if m_num > num: 
                num = m_num
                full_model_name = i
    if num != 0:
        model_names.append((network_name, full_model_name))
     

for ix, (net_name, model_file) in enumerate(model_names):
    print('\nworking on {}...'.format(net_name))
    print('model name: {}'.format(model_file))
    #print('Output label: {}'.format(outs[ix]))
    m_eval = ModelEval(model_file,
                       train_dirs=train_dirs,
                       test_dirs=test_dirs,
                       labels_store=labels_store)
    save_dir = model_dirs[ix]
    print('will save in {}'.format(save_dir))
    
    cube_clip = .16
    pot_clip = .6

    # !!!!!!!!!!!!!!!!!!!!!!!!!
    #rewriting for this run....
    scale_cube = 1./cube_clip
    scale_pot = 1./pot_clip

    meta={'cube_scale':scale_cube,
          'pot_scale':scale_pot,
          'cube_clip':cube_clip,
          'pot_clip':pot_clip}

    m_eval.act_vs_pred(save_dir, meta=meta)

