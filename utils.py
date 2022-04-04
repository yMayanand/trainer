from torch.utils.tensorboard import SummaryWriter

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
