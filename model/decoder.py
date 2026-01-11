from torch import cat, nn, sum, stack
from torch.nn.functional import pad
from model.encoder import Encoder
#from model.skeleton_unpool import SkeletonUnpool
from model.transformer_selfattention import TransformerBlock
from config.manifest import get_manifest
from utils.data_manager import DataManager

DM = DataManager()

class Decoder(nn.Module):
    def __init__(self, encoder: Encoder):
        super(Decoder, self).__init__()
        _MANIFEST = get_manifest()
        self._window_size = _MANIFEST.DATA.WINDOW_SIZE
        self.encoder = encoder # stuff in here we need later (in Decoder.forward)
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self._skeleton_info = _MANIFEST.DATA.SKELETON_INFO_HANDLING
        self._rotation_type = _MANIFEST.DATA.ROTATION_TYPE
        self.convolutions = []

        _number_of_layers = _MANIFEST.MODEL.NUMBER_OF_LAYERS
        _kernel_size = _MANIFEST.TRAINING.KERNEL_SIZE
        _padding = (_kernel_size - 1) // 2

        _add_offset = self._skeleton_info == 'concat'

        # the decoder takes latent motion data, and positional data, and creates output motion data
        self._window_size = _MANIFEST.DATA.WINDOW_SIZE  # used in forward function
        _LATSZ = _MANIFEST.MODEL.LATSZ

        out_md_sz = (encoder.channel_base[0]) * encoder.edges_counts[0] # decoder output_motion_data_size
        self.poslen = (encoder.channel_base[0]-1) * encoder.edges_counts[0]
        self.layers.append(TransformerBlock(self._window_size, 2*out_md_sz, oldk=_LATSZ+self.poslen, heads=encoder.edges_counts[0]))
        self.layers.append(TransformerBlock(self._window_size, 2*out_md_sz, oldk=2*out_md_sz, heads=encoder.edges_counts[0]))
        self.layers.append( nn.Linear( 2*out_md_sz*self._window_size, out_md_sz ) ) # to map all vectors down to 1

    def forward(self, decoder_input, offset=None):
        '''forward function for decoder'''

        if self._skeleton_info == 'concat' and offset is not None:
            bs = offset[0].shape[0] # take *some* of the offset data
            offpad = offset[0].view(bs,-1) # view it as a single vector
            offpad = offpad.unsqueeze(-1).expand((-1, -1, decoder_input.shape[-1])) # expand it (repeating along time)
            decoder_input = cat((decoder_input, offpad), dim=-2)  # glue it on to the motion data axis
        else:
            raise NotImplementedError('Decoder: alt conditional in forward function hit')

        # we transform the latent motion data into output motion representation
        # pad the latent motion data on either side of the time axis 
        _MANIFEST = get_manifest()
        _LATSZ = _MANIFEST.MODEL.LATSZ
        out_md_sz = self.encoder.channel_base[0] * self.encoder.edges_counts[0]
        padleftsz = self._window_size//2
        padrightsz = self._window_size - padleftsz
        pad_input = pad( decoder_input, (padleftsz, padrightsz), 'replicate' )
        # for each frame in latent motiondata input, 
        # grab 'window_sz' frames from (padded) input, and turn it into a single output frame of motion data
        out = []
        for i in range( decoder_input.shape[-1] ): 
            x = pad_input[:,:,i:i+self._window_size] 
            x = x.transpose(-1,-2)  # attention wants numvecs in dim-2, veclen in dim-1
            x = self.layers[0](x)
            x = x + self.layers[1](x) 
            x = x.view( (bs, 2*self._window_size*out_md_sz) )
            out.append( self.layers[2](x) )
        result= stack( out, dim=-1 )

        if self._rotation_type == 'quaternion':
            result = result[:, :-1, :]

        return result
