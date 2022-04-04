class LearningRateMonitor:
    def __init__(self, logging_interval=None, log_momentum=False):
        self.log_momentum = log_momentum
        self.logging_interval = logging_interval

    def on_train_batch_start(self, trainer, model):
        if self.logging_interval == 'step':
            trainer.writer.add_scalar(
                'lr', trainer.optimizer.param_groups[0]['lr'], trainer.global_step)
        elif self.logging_interval == None:
            if trainer.global_step % trainer.sched_dict['frequency'] == 0:
                trainer.writer.add_scalar(
                    'lr', trainer.optimizer.param_groups[0]['lr'], trainer.global_step)

    def on_train_epoch_start(self, trainer, model):
        if self.logging_interval == 'epoch':
            trainer.writer.add_scalar(
                'lr', trainer.optimizer.param_groups[0]['lr'], trainer.global_step)
