from config.manifest import get_manifest
from torch import device, norm
import torch.nn as nn

class Criterion_EE:
    def __init__(self, base_criterion, norm_eps=0.008):
        self._base_criterion = base_criterion
        self._norm_eps = norm_eps

    def __call__(self, pred, gt):
        _MANIFEST = get_manifest()
        reg_ee_loss = self._base_criterion(pred, gt)
        extra_ee_loss = 0

        if _MANIFEST.DATA.EE_VELO:
            gt_norm = norm(gt, dim=-1)
            contact_index = gt_norm < self._norm_eps
            extra_ee_loss = self._base_criterion(pred[contact_index], gt[contact_index])
            if extra_ee_loss.isnan():
                extra_ee_loss = 0

        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return []

class Criterion_EE_2:
    '''
    Alternative version presumably. This one is defined as adaptive ee
    '''
    def __init__(self, base_criterion, norm_eps=0.008):
        _MANIFEST = get_manifest()
        self._base_criterion = base_criterion
        self._norm_eps = norm_eps
        self._ada_para = nn.Linear(15,15).to(device(_MANIFEST.SYSTEM.DEVICE))

    def __call__(self, pred, gt):
        pred = pred.reshape(pred.shape[:-2]+(-1,))
        gt = gt.reshape(gt.shape[:-2]+(-1,))
        pred = self._ada_para(pred)
        reg_ee_loss = self._base_criterion(pred, gt)
        extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return list(self.ada_para.parameters())
