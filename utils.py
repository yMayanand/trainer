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
