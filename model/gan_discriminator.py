from torch import nn, sigmoid, transpose
from config.manifest import get_manifest
#from model.skeleton_convolution import SkeletonConvolution
from model.skeleton_pool import SkeletonPool
from model.transformer_selfattention import TransformerBlock
from utils.data_manager import DataManager

DM = DataManager()

class Discriminator(nn.Module):
    def __init__(self, edges, skeleton_type):
        super(Discriminator, self).__init__()
        _MANIFEST = get_manifest()
        kernel_size = _MANIFEST.TRAINING.KERNEL_SIZE
        number_of_layers = _MANIFEST.MODEL.NUMBER_OF_LAYERS
        degree_of_separation = _MANIFEST.TRAINING.DEGREE_OF_SEPARATION
        _window_size = _MANIFEST.DATA.WINDOW_SIZE

        self.topologies = [edges]
        self.pooling_list = []
        self.number_of_joints = [len(edges) + 1]

        self._is_patch_gan = _MANIFEST.MODEL.IS_PATCH_GAN

        self._channel_base = self.__setup_channel_base__(number_of_layers)
        self._layers = nn.ModuleList()

        self.channel_list, self._last_channel = self.__setup_channel_list__(number_of_layers, kernel_size, edges, degree_of_separation, skeleton_type)

        if not self._is_patch_gan:
            self.compress = nn.Linear(in_features=self._last_channel, out_features=1)

        self._layers.append(TransformerBlock(_window_size, self._last_channel*9, oldk=self.channel_list[0],heads=1))

    @staticmethod
    def __setup_channel_base__(number_of_layers):
        channel_base = [3]
        for i in range(number_of_layers):
            channel_base.append(channel_base[-1] * 2)
        return channel_base

    def __setup_channel_list__(self, number_of_layers, kernel_size, edges, degree_of_separation, skeleton_type):
        channel_list = []
        last_channel = 0
        padding = (kernel_size - 1) // 2

        for i in range(number_of_layers):
            sequence = []
            neighbour_list = DM.find_neighbours(self.topologies[i], degree_of_separation)
            channels_in = self._channel_base[i] * self.number_of_joints[i]
            if i == 0:
                channel_list.append(channels_in)

            channels_out = self._channel_base[i+1] * self.number_of_joints[i]
            channel_list.append(channels_out)

            if i < number_of_layers - 1:
                bias = False
            else:
                bias = True

            if i == number_of_layers - 1:
                kernel_size = 16
                padding = 0

            joint_number = self.number_of_joints[i]

            if i < number_of_layers - 1:
                sequence.append(nn.BatchNorm1d(channels_out))

            channels_per_edge = channels_out // len(neighbour_list)

            pool = SkeletonPool(edges=edges, channels_per_edge=channels_per_edge,
                                layer_number=i, skeleton_type=skeleton_type, last_pool=False)
            sequence.append(pool)

            if not self._is_patch_gan or i < number_of_layers - 1:
                sequence.append(nn.LeakyReLU(negative_slope=0.2))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.number_of_joints.append(len(pool.new_edges)+1)

            if i == number_of_layers - 1:
                last_channel = self.number_of_joints[-1] * self._channel_base[i+1]

        return channel_list, last_channel

    def forward(self, d_input):
        '''

        :param d_input: discriminator input
        :return:
        '''
        d_input = d_input.reshape(d_input.shape[0], d_input.shape[1], -1)
        d_input = d_input.permute(0, 2, 1)
        original_shape = d_input.shape
        d_input = transpose(d_input,-1,-2)

        for layer in self._layers:
            d_input = layer(d_input)

        if not self._is_patch_gan:
            d_input = d_input.reshape(d_input.shape[0], -1)
            d_input = self.compress(d_input)

        d_input = d_input.sum(-2)
        d_input = d_input.view(original_shape[0], self._last_channel, 9)

        return sigmoid(d_input).squeeze()
