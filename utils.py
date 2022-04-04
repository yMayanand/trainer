def get_basic_writer(log_dir='runs/exp', comment=''):
    writer = SummaryWriter(log_dir=log_dir, comment=comment)
    layout = { "Trainig Metrics": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
        },
    }
    writer.add_custom_scalars(layout)
    return writer
