from config.manifest import get_manifest
from model.skeleton_linear import SkeletonLinear
from model.skeleton_pool import SkeletonPool
from utils.data_manager import DataManager
from torch import nn, Size, Tensor

DM = DataManager()

class StaticEncoder(nn.Module):
    '''
    Encoder for the 'static' part i.e. offsets
    '''
    def __init__(self, edges,skeleton_type):
        super(StaticEncoder, self).__init__()
        _MANIFEST = get_manifest()
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(2): # note this '2' is originally args.number_of_layers
            neighbour_list = DM.find_neighbours(edges, _MANIFEST.TRAINING.DEGREE_OF_SEPARATION)
            sequence = []
            sequence.append(SkeletonLinear(neighbours=neighbour_list,
                                           channels_in=channels*len(neighbour_list),
                                           channels_out=channels*len(neighbour_list)*2,
                                           extra_dimension=True))

            if i < 2 - 1:
                pool = SkeletonPool(edges=edges, channels_per_edge=channels*2,
                                    layer_number=i,skeleton_type=skeleton_type, last_pool=False)
                sequence.append(pool)
                edges = pool.new_edges
            sequence.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*sequence))

    def forward(self, input: Tensor):
        '''
        :input Tensor: x y z
                where x = number of characters in the group
                y = number of bones
                z = position vector for bone
        :output
        '''
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))
        if not(output[0].shape==Size([4,23,3]) or output[0].shape==Size([4,28,3])):
            pass
        return output