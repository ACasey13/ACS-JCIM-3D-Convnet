import keras
import numpy as np
import os
import pandas as pd

class LossHistory(keras.callbacks.Callback):
    def __init__(self, m_dir):
        self.efile = os.path.join(m_dir, 'epoch_history.csv')
        #self.bfile = os.path.join(m_dir, 'batch_history.txt')

    #def on_train_begin(self, logs={}):
        #self.log = {k:[] for k in logs.keys()}
        

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 0:
            #print(logs)
            self.log = {k:[] for k in logs.keys()}
        else:
            [self.log[k].append(v) for k,v in logs.items()]
        df = pd.DataFrame(self.log)
        df.to_csv(self.efile, mode='w')
  
    #def on_batch_end(self, batch, logs={}):
    #    self.loss.append((batch, logs.get('loss'), logs.get('mean_absolute_error')))
    #    arr = np.array(self.loss)
    #    np.savetxt(self.bfile, arr)

