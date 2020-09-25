import keras
print('Keras version: {}\n'.format(keras.__version__))
import sys
import os
#append utils directory to PYTHONPATH if not done so already
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

if os.path.abspath('../utils') not in [os.path.abspath(path) for path in user_paths]:
    sys.path.append(os.path.abspath('../utils/'))

from model_modules import naive_inception as ni
from model_modules import vanilla_multi as vanilla
from model_modules import vanilla_multi_2 as vanilla_2
from model_modules import threeDIncResNet_multi_output_model as iresnet
from keras.utils import plot_model

graph_dir = os.path.join('.','models_graphs')

models = {'naive_inception': ni(), 
          'vanilla': vanilla(),
          'vanilla_2': vanilla_2(),
          'iresnet': iresnet()}

for k,v in models.items():
    print('graphing model {}...'.format(k))
    plot_model(v, show_shapes=True, show_layer_names=False,
               to_file=os.path.join(graph_dir, '{}.pdf'.format(k)))

print('finished graphing models!')

