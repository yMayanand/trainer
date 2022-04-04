import os
import torch
from pathlib import Path
import math

from sklearn import model_selection


class ModelCheckpoint:
    def __init__(self, mode='min', monitor=None,
                 save_only_weights=False):
        self.dir_path = Path('./runs/checkpoints')
        self.dir_path.mkdir(exist_ok=True)
        self.mode = mode
        self.monitor = monitor
        self.save_only_weights = save_only_weights
        self.prev_fname = None
        self.metric_val = -math.inf if mode == 'max' else math.inf

    def on_train_epoch_end(self, trainer, model):
        fname = f"{trainer.name}_{trainer.global_step}.pt"
        fname = os.path.join(self.dir_path, fname)
        def save():
            if self.prev_fname:
                os.remove(self.prev_fname)
            if self.save_only_weights:
                torch.save(model.state_dict(), fname)
            else:
                torch.save({'model_state': model.state_dict(),
                            'opt_state': trainer.optimizer.stat_dict()}, fname)
        if self.monitor:
            temp = trainer.result[self.monitor]
            if self.mode == 'max':
                if temp > self.metric_val:
                    self.metric_val = temp
                    save()
            elif self.mode == 'min':
                if temp < self.metric_val:
                    self.metric_val = temp
                    save()
        else:
            save()
        self.prev_fname = fname
