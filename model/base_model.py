from abc import ABC, abstractmethod
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, MultiStepLR

from config.manifest import get_manifest
from utils.loss_recorder import LossRecorder
import os

class BaseModel(ABC):
    list_of_schedulers = ['default','linear', 'Step_LR', 'Plateau', 'MultiStep']

    def __init__(self):
        self.schedulers = []
        self.optimizers = []
        self.epoch_count = 0

        if get_manifest().TRAINING.IS_TRAINING:
            log_path = get_manifest().MODEL.LOGS_DIR
            self.writer = SummaryWriter(log_dir=log_path)
            self.loss_recorder = LossRecorder(self.writer)

    @abstractmethod
    def set_input(self, input_data: dict, is_training: bool):
        pass

    @abstractmethod
    def _compute_test_result(self):
        pass

    @abstractmethod
    def _forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self, is_training: bool, device: str, number_of_layers: int, ee_velo: bool, ee_from_root: bool, gan_mode: str):
        pass

    def _get_scheduler(self, optimizer):
        _MANIFEST = get_manifest()
        scheduler = _MANIFEST.MODEL.SCHEDULER

        if scheduler not in self.list_of_schedulers:
            raise NotImplementedError(
                'BaseModel.get_schedulers does not support {} scheduler type.'.format(scheduler))

        if scheduler == 'default':
            return

        if scheduler == 'linear':
            learning_rate = lambda epoch: 1.0 - max(0, epoch - _MANIFEST.MODEL.N_EPOCHS_ORIGIN) / float(_MANIFEST.MODEL.N_EPOCHS_DECAY + 1)
            return LambdaLR(optimizer, learning_rate)

        if scheduler == 'Step_LR':
            return StepLR(optimizer, step_size=_MANIFEST.MODEL.LR_STEP_SIZE, gamma=_MANIFEST.MODEL.LR_GAMMA)

        if scheduler == 'Plateau':
            return ReduceLROnPlateau(optimizer, mode=_MANIFEST.MODEL.LR_PLAT_MODE, factor=_MANIFEST.MODEL.LR_PLAT_FACTOR,
                                     threshold=_MANIFEST.MODEL.LR_PLAT_THRESHOLD, patience=_MANIFEST.MODEL.LR_PLAT_PATIENCE,
                                     verbose=_MANIFEST.MODEL.LR_PLAT_VERBOSE)
        if scheduler == 'MultiStep':
            return MultiStepLR(optimizer, milestones=_MANIFEST.MODEL.LR_MULTISTEP_MILESTONES)

    def setup(self):
        if get_manifest().TRAINING.IS_TRAINING:
            self.schedulers = [self._get_scheduler(optimizer) for optimizer in self.optimizers]

    def epoch(self):
        self.loss_recorder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        self.epoch_count += 1

    def test(self):
        with no_grad():
            self._forward()
            self._compute_test_result()