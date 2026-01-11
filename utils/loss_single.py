from torch.utils.tensorboard import SummaryWriter
import numpy as np

class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer

    def add_scalar(self, scalar_value, step=None):
        if step is None:
            step = len(self.loss_epoch)
        self.loss_step.append(scalar_value)
        self.loss_epoch_tmp.append(scalar_value)
        self.writer.add_scalar('Train/step_' + self.name, scalar_value, step)

    def epoch(self, step=None):
        if step is None:
            step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)