import random

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm

from .utils import get_basic_writer, create_layout


class Trainer:
    def __init__(self, num_epochs=1, writer=None,
                 mixed_precision=False, cbs=[],
                 tensorboard_metrics=['loss', 'accuracy'],
                 name='exp0'):
        
        log_dir = 'runs/'
        layout = create_layout(tensorboard_metrics)
        if writer is None:
            writer = get_basic_writer(layout, log_dir=log_dir+name)
        self.amp_ = mixed_precision
        self.writer = writer
        self.num_epochs = num_epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 1
        self.cbs = cbs
        self.available_cbs = ['on_train_epoch_start',
                              'on_train_epoch_end',
                              'on_val_epoch_start',
                              'on_val_epoch_end',
                              'on_fit_start',
                              'on_fit_end',
                              'on_train_batch_start',
                              'on_train_batch_end',
                              'on_val_batch_start',
                              'on_val_batch_end']

    def noop(x=None, *args, **kwargs):
        """this function does nothing just returns input as it is"""
        return x

    def __call__(self, event):
        if event in self.available_cbs:
            for cb in self.cbs:
                getattr(cb, event, self.noop)(self)

    def init_optimizers(self, model):
        temp = model.configure_optimizers()
        if isinstance(temp, (tuple, list)):
            if len(temp) == 1:
                *self.optimizer, = temp
            elif len(temp) > 1:
                self.optimizer, self.sched_dict = temp
        elif isinstance(temp, torch.optim.Optimizer):
            self.optimizer = temp
            self.sched_dict = None
        else:
            raise ValueError('correct optimizer not provided')

    def generate_colors(self, num):
        colors = []
        for i in range(num):
            R = random.randint(0, 254)
            G = random.randint(0, 254)
            B = random.randint(0, 254)
            colors.append(f'\033[38;2;{R};{G};{B}m')
        self.colors = colors

    def fit(self, model, train_dl=None, val_dl=None):
        self('on_fit_start')
        if self.device == 'cuda':
            self.scaler = GradScaler()
        model.to(self.device)
        model.writer = self.writer
        self.init_optimizers(model)
        self.generate_colors(len(model.log_metric))
        assert train_dl != None, "training could not proceed as train_dl not provided"
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            self.train_loop(model, train_dl)
            if val_dl is not None:
                self.validation_loop(model, val_dl)
            if self.sched_dict is not None:
                if self.sched_dict['strategy'] == 'epoch':
                    if self.sched_dict['monitor_metric'] is not None:
                        temp = self.sched_dict['monitor_metric']
                        metric = getattr(model, temp, None)
                        assert metric != None, f"monitor_metric {temp} not recorded"
                        self.sched_dict['scheduler'].step(metric.avg)
            logs = self.print_factory(model)
            print(logs)
            self.reset_metrics(model)
            self('on_fit_end')

    def reset_metrics(self, model):
        for i in model.log_metric:
            metric = getattr(model, i, None)
            if metric is not None:
                metric.reset()

    def print_factory(self, model):
        def template(metric, val, color):
            return color + '\033[1m' + f"{metric}:\u001b[30m{val.avg: .4f}"

        temp = []
        for metric, color in zip(model.log_metric, self.colors):
            temp.append(template(metric, getattr(model, metric, ''), color))
        return ' '.join(temp)

    def to_device(self, val, tmp=None):
        if tmp is None:
            tmp = []
        if isinstance(val, (list, tuple)):
            for i in val:
                self.to_device(i, tmp=tmp)
        elif isinstance(val, torch.Tensor):
            tmp.append(val.to(self.device))
        return tmp

    def train_with_amp(self, model, batch, batch_idx):
        if self.device != 'cuda':
            raise ValueError('cuda device not found')
        with autocast():
            loss = model.training_step(batch, batch_idx)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_loop(self, model, dl):
        self('on_train_epoch_start')
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc='Training...'):
            self('on_train_batch_start')
            model.global_step = self.global_step
            model.train()
            self.optimizer.zero_grad()
            batch = self.to_device(batch)
            if self.amp_:
                self.train_with_amp(model, batch, batch_idx)
            else:
                loss = model.training_step(batch, batch_idx)
                loss.backward()
                self.optimizer.step()
            if self.sched_dict is not None:
                if self.sched_dict['strategy'] == 'step':
                    if self.global_step % self.sched_dict['frequency'] == 0:
                        self.sched_dict['scheduler'].step()
                        #self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self('on_train_batch_end')
            self.global_step += 1
        self('on_train_epoch_end')

    def validation_loop(self, model, dl):
        self('on_val_epoch_start')
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc='Validating...'):
            self('on_val_batch_start')
            model.eval()
            with torch.no_grad():
                batch = self.to_device(batch)
                loss = model.validation_step(batch, batch_idx)
            self('on_val_batch_end')
        self('on_val_epoch_end')

    def predict(self, model, dl):
        return self.predict_loop(model, dl)

    def predict_loop(self, model, dl):
        temp = []
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc='Validating...'):
            model.eval()
            with torch.no_grad():
                batch = self.to_device(batch)
                val = model.predict_step(batch, batch_idx)
                temp.append(val)

        return temp
