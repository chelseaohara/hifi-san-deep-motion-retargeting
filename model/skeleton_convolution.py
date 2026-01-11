from torch import nn, zeros, zeros_like
from model.skeleton_linear import SkeletonLinear
from math import sqrt

'''Refactored SkeletonConvolution Class, now bypassed by use of Transformer block'''

class SkeletonConvolution(nn.Module):
    valid_padding_modes = ['constant', 'reflect']
    def __init__(self, neighbours, channels_in, channels_out, joint_number, kernel_size, is_bias, padding_mode, padding, stride, add_offset):
        '''
        :param neighbours:
        :param channels_in:
        :param channels_out:
        :param joint_number:
        :param kernel_size:
        :param is_bias:
        :param padding_mode:
        :param padding:
        :param stride:
        :param add_offset:
        '''
        super(SkeletonConvolution, self).__init__()
        if channels_in % joint_number != 0 or channels_out % joint_number != 0:
            raise ValueError("channels_in and channels_out must be divisible by the number of joints")

        if padding_mode not in self.valid_padding_modes:
            raise ValueError("padding_mode must be one of: {}".format(self.valid_padding_modes))

        self._channels_in_per_joint = channels_in // joint_number
        self._channels_out_per_joint = channels_out // joint_number
        self._padding_mode = padding_mode
        self._padding = padding
        self._stride = stride
        self._dilation = 1
        self._groups = 1

        self._expanded_neighbours = self.__get_expanded_neighbours__(neighbours)

        self.offset = None
        self._add_offset = add_offset

        if self._add_offset:
            self._offset_encoder = SkeletonLinear(neighbours, channels_in, channels_out)

        self.weight = zeros(channels_out, channels_in, kernel_size)

        if is_bias:
            self.bias = zeros(channels_out)
        else:
            self.register_parameter('bias', None)

        self.mask = self.__get_mask__()

        self._reset_parameters()

    def __get_expanded_neighbours__(self, neighbours):
        expanded_neighbours = []
        for neighbour in neighbours:
            expanded = []
            for k in neighbour:
                for i in range(self._channels_in_per_joint):
                    expanded.append(k * self._channels_in_per_joint + i)
            expanded_neighbours.append(expanded)

        return expanded_neighbours

    def __get_mask__(self):
        mask = zeros_like(self.weight)
        for i, neighbour in enumerate(self._expanded_neighbours):
            mask[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1), neighbour, ...] = 1

        mask = nn.Parameter(mask, requires_grad=False)

        return mask

    def _reset_parameters(self):
        for i, neighbour in enumerate(self._expanded_neighbours):
            weight_slice = zeros_like(
                self.weight[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1), neighbour, ...])
            nn.init.kaiming_uniform_(weight_slice, a=sqrt(5))
            self.weight[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1), neighbour,
            ...] = weight_slice

            if self.bias is not None:
                fan_in, nothing = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1), neighbour,
                    ...])
                bound = 1 / sqrt(fan_in)
                weight_slice = zeros_like(
                    self.bias[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1)])
                nn.init.uniform_(weight_slice, -bound, bound)
                self.bias[self._channels_out_per_joint * i: self._channels_out_per_joint * (i + 1)] = weight_slice

        self.weight = nn.Parameter(self.weight)

        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, new_offset):
        if not self._add_offset:
            raise Exception('SkeletonConvolution._add_offset is False: wrong combination of parameters?')
        self.offset = new_offset.reshape(new_offset.shape[0], -1)

    def forward(self, input):
        weight_mask = self.weight * self.mask
        result = nn.functional.conv1d(nn.functional.pad(input, (self._padding, self._padding), mode=self._padding_mode), weight_mask,
                                      self.bias, self._stride, 0, self._dilation, self._groups)

        if self._add_offset:
            offset_result = self._offset_encoder(self.offset)
            offset_result = offset_result.reshape(offset_result.shape + (1,))
            result += offset_result/100

        return result