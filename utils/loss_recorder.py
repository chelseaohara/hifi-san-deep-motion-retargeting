from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from utils.loss_single import SingleLoss

class LossRecorder:
    def __init__(self, writer: SummaryWriter):
        self.losses = {}
        self.writer = writer

    def add_scalar(self, name, scalar_value, step=None):
        if isinstance(scalar_value, Tensor):
            scalar_value = scalar_value.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer)
        self.losses[name].add_scalar(scalar_value, step)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)