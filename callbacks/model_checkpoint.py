import os

from sklearn import model_selection

class ModelCheckpoint:
    def __init__(self, mode='min', monitor=None,
                 save_only_weights=False):
        self.dir_path = 'runs'
        self.mode = mode
        self.monitor = monitor
        self.save_only_weights = save_only_weights

    #def on_train_epoch_end(self, model)
        

