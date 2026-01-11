from torch import device, nn, Tensor, zeros
from torch.nn import functional
from config.manifest import get_manifest

class SkeletonLinear(nn.Module):
    def __init__(self, channels_in, channels_out, extra_dimension=False):
        super(SkeletonLinear, self).__init__()

        self._device = get_manifest().SYSTEM.DEVICE
        self._extra_dimension = extra_dimension

        self.weight = zeros(channels_out, channels_in)
        self.mask = zeros(channels_out, channels_in)
        self.bias = nn.Parameter(Tensor(channels_out))

    def forward(self, module_input):
        module_input = module_input.reshape(module_input.shape[0], -1)
        weight_masked = self.weight * self.mask
        result = functional.linear(module_input.to(device(self._device)), weight_masked.to(device(self._device)), self.bias).to(device(self._device))
        if self._extra_dimension:
            result = result.reshape(result.shape + (1,))
        return result