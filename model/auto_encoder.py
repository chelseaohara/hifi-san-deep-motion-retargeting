from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, edges, skeleton_type):
        # edges is bone_details_list, list of tuples containing information
        # about simplified bones (parent, child, offset)
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(edges, skeleton_type)
        self.decoder = Decoder(self.encoder)

    def forward(self, skeleton_data, offset):
        '''
        :param skeleton_data: Tensor (m x n x o)
        :param offset: list of Tensors (m x n x o)
        :return: tuple: latent, result
        '''
        latent = self.encoder(skeleton_data, offset)
        result = self.decoder(latent, offset)
        return latent, result