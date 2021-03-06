import random
import copy


import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm

from .utils import *
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, num_epochs=1, writer=None,
                 mixed_precision=False, cbs=[],
                 track_metrics=['loss', 'accuracy'],
                 name='exp0',
                 set_to_none=False):

        log_dir = 'runs/'
        self.name = name
        self.set_to_none = set_to_none
        layout = create_layout(track_metrics)
        if writer is None:
            writer = get_basic_writer(layout, log_dir=log_dir+self.name)
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

    def __call__(self, event, model):
        if event in self.available_cbs:
            for cb in self.cbs:
                getattr(cb, event, self.noop)(self, model)

    def lr_finder(self, model, train_dl=None,
                  min_val=-7, max_val=1,
                  iters=100, smooth=0.95):
        model.global_step = self.global_step
        lrs = torch.logspace(min_val, max_val, steps=iters)
        if train_dl is None:
            raise MisConfigurationError('train_dl not provided')

        it = iter(train_dl)
        model = copy.deepcopy(model)
        model.to(self.device)
        self.init_optimizers(model)
        losses = []
        for batch_idx in tqdm(range(iters), leave=False):
            self.optimizer.param_groups[0]['lr'] = lrs[batch_idx]
            batch = next(it)
            batch = self.to_device(batch)
            self.optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            self.optimizer.step()
            if batch_idx > 0:
                losses.append(loss.item())
            if batch_idx == (len(train_dl) - 1):
                it = iter(train_dl)
        loss = model.training_step(batch, batch_idx)
        losses.append(loss.item())
        smoothed_losses = smoother(losses, smooth)
        fig, ax = plt.subplots()
        ax.set_title('LR Finder Plot')
        ax.plot(lrs, smoothed_losses)
        ax.set_xscale('log')
        return fig, ax
        


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
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise MisConfigurationError('correct optimizer not provided')

    def generate_colors(self, num):
        colors = []
        for i in range(num):
            R = random.randint(0, 254)
            G = random.randint(0, 254)
            B = random.randint(0, 254)
            colors.append(f'\033[38;2;{R};{G};{B}m')
        self.colors = colors

    def fit(self, model, train_dl=None,
            val_dl=None, seed=42):
        seed_everything(seed=seed)
        self('on_fit_start', model)
        if self.device.type == 'cuda' and self.amp_:
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
            self.result = self.create_results(model)
            self.reset_metrics(model)
            self('on_train_epoch_end', model)
        self('on_fit_end', model)
        return self.result

    def reset_metrics(self, model):
        for i in model.log_metric:
            metric = getattr(model, i, None)
            if metric is not None:
                metric.reset()

    def create_results(self, model):
        def get_val(metric):
            val = getattr(model, metric, None)
            return val.avg
        metric_dict = {}
        for i in model.log_metric:
            metric_dict.update({i: get_val(i)})
        return metric_dict

    def print_factory(self, model):
        def template(metric, val, color):
            return color + '\033[1m' + f"{metric}:\u001b[30m{val.avg: .4f}"

        temp = []
        for metric, color in zip(model.log_metric, self.colors):
            temp.append(template(metric, getattr(model, metric, None), color))
        return ' '.join(temp)

    def to_device(self, val):
        tmp = []
        if isinstance(val, torch.Tensor):
            return val.to(self.device)
        elif isinstance(val, (tuple, list)):
            for i in val:
                tmp.append(self.to_device(i))
            return tmp

    def train_with_amp(self, model, batch, batch_idx):
        if self.device.type != 'cuda':
            raise MisConfigurationError('cuda device not found')
        with autocast():
            loss = model.training_step(batch, batch_idx)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_loop(self, model, dl):
        self('on_train_epoch_start', model)
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc='Training...'):
            self('on_train_batch_start', model)
            model.global_step = self.global_step
            model.train()
            self.optimizer.zero_grad(set_to_none=self.set_to_none)
            batch = self.to_device(batch)
            if self.amp_:
                self.train_with_amp(model, batch, batch_idx)
            else:
                loss = model.training_step(batch, batch_idx)
                loss.backward()
                self.optimizer.step()
            if self.sched_dict is not None:
                if self.sched_dict['strategy'] == 'step':
                    if (batch_idx+1) % self.sched_dict['frequency'] == 0:
                        self.sched_dict['scheduler'].step()
                        #self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self('on_train_batch_end', model)
            self.global_step += 1

    def validation_loop(self, model, dl):
        self('on_val_epoch_start', model)
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc='Validating...'):
            self('on_val_batch_start', model)
            model.eval()
            with torch.no_grad():
                batch = self.to_device(batch)
                loss = model.validation_step(batch, batch_idx)
            self('on_val_batch_end', model)
        self('on_val_epoch_end', model)

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
