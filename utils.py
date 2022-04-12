from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch

# for reference
layout = {"Training_Metrics": {
    "loss": ["Multiline", ["loss/train", "loss/val"]],
    "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
},
}


def create_layout(metrics):
    temp = {}
    for metric in metrics:
        temp.update(
            {metric: ["Multiline", [f"{metric}/train", f"{metric}/val"]]})
    layout = {'training_metrics': temp}
    return layout


def get_basic_writer(layout, log_dir='runs/exp', comment=''):
    writer = SummaryWriter(log_dir=log_dir, comment=comment)
    writer.add_custom_scalars(layout)
    return writer

class MisConfigurationError(Exception):
    pass

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum/self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0


def smoother(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

