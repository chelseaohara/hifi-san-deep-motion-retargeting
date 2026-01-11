from torch import nn, tensor
from config.manifest import get_manifest

class GANLoss(nn.Module):
    def __init__(self, gan_mode, real_label = 1.0, fake_label = 0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', tensor(real_label))
        self.register_buffer('fake_label', tensor(fake_label))

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

    def __call__(self, prediction, target_is_real):

        _MANIFEST = get_manifest()
        device = _MANIFEST.SYSTEM.DEVICE

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        target_tensor_expanded = target_tensor.expand_as(prediction)
        loss = self.loss(prediction, target_tensor_expanded.to(device))

        return loss