from torch import cat, nn, zeros_like, stack, sum
from torch.nn.functional import pad 
from config.manifest import get_manifest
#from model.skeleton_convolution import SkeletonConvolution
#from model.skeleton_pool import SkeletonPool
from model.transformer_selfattention import TransformerBlock
from utils.data_manager import DataManager

DM = DataManager()
_MANIFEST = get_manifest()

class Encoder(nn.Module):
    def __init__(self, edges, skeleton_type):
        super(Encoder, self).__init__()

        self._device_type = _MANIFEST.SYSTEM.DEVICE
        self._number_of_layers = _MANIFEST.MODEL.NUMBER_OF_LAYERS
        self._rotation_type = _MANIFEST.DATA.ROTATION_TYPE

        # used by decoder
        self.channel_base = self.__get_channel_base__()

        self._position_representation = _MANIFEST.DATA.POSITION_REPRESENTATION


        _kernel_size = _MANIFEST.TRAINING.KERNEL_SIZE
        _padding = (_kernel_size - 1) // 2
        _bias = True

        self._skeleton_info_handling = _MANIFEST.DATA.SKELETON_INFO_HANDLING # used in forward() function
        _add_offset = self._skeleton_info_handling == 'concat'

        # used by decoder!
        self.topologies = [edges] # wrapping the list of bone details in a list because simplified details will be appended in loop below
        self.pooling_list = []
        self.channels_list = []

        # this list (originally called edge_num) starts with one value for the length of edges + 1
        # and gets appended to for the second layer with a smaller value, assuming a simplified # of edges
        # NOTE: this list is used by the decoder
        self.edges_counts = [len(edges)+1]

        self.layers = nn.ModuleList()

        self.convolutions = [] # this is the list to build for the forward function

        # the encoder takes input motion data, and positional data, and creates latent motion data
        self._window_size = _MANIFEST.DATA.WINDOW_SIZE # used in forward function
        self.poslen = (self.channel_base[0]-1) * self.edges_counts[0] 
        self.motdatlen = (self.channel_base[0]) * self.edges_counts[0] 
        _LATSZ = _MANIFEST.MODEL.LATSZ
        _oldk = self.motdatlen + self.poslen  # length of input_motion+positional vectors

        self.layers.append(TransformerBlock(self._window_size, _LATSZ, oldk=_oldk, heads=_LATSZ//2))
        self.layers.append(TransformerBlock(self._window_size, _LATSZ, heads=_LATSZ//2))
        self.layers.append( nn.Linear( _LATSZ*(self._window_size), _LATSZ ) ) # to map all vectors down to 1

    def __get_channel_base__(self):
        if self._rotation_type == 'quaternion':
            channel_base = 4
        elif self._rotation_type == 'euler_angle':
            channel_base = 3
        else:
            raise NotImplementedError('Encoder: Rotation type {} from Manifest is not supported. '
                                      'Must be `quaternion` or `euler` rotation.'.format(self._rotation_type))

        channel_base_list = [channel_base]

        for i in range(self._number_of_layers):
            # append last doubled
            channel_base_list.append(channel_base_list[-1] * 2)

        return channel_base_list

    def forward(self, encoder_input, offset=None):
        if self._rotation_type == 'quaternion' and self._position_representation != '4D':
            encoder_input = cat((encoder_input,zeros_like(encoder_input[:,[0],:])), dim=1)

        if self._skeleton_info_handling == 'concat' and offset is not None:
            bs = offset[0].shape[0]
            offpad = offset[0].view(bs,-1) # view offset data[0] as a single vector (for each batch)
            offpad = offpad.unsqueeze(-1).expand((-1,-1,encoder_input.shape[-1])) # expand it for gluing (expand in time)
            # now glue a copy of the static data onto movement data for *each* timeframe
            encoder_input = cat((encoder_input.to(_MANIFEST.SYSTEM.DEVICE), offpad.to(_MANIFEST.SYSTEM.DEVICE)),dim=-2)
        else:
            raise Warning('Encoder: alt to conditional hit')

        _LATSZ = _MANIFEST.MODEL.LATSZ
        # we transform the motion data into 'latent' representation
        # pad the motion data on either side of the time axis 
        # then grab window_size frames at a time and convert to a single output frame
        padleftsz = self._window_size//2
        padrightsz = self._window_size - padleftsz
        # we pad left and right of input motion data with copies of the starting and ending frame
        pad_input = pad( encoder_input, (padleftsz, padrightsz), 'replicate' )
        out = [] # append completed output frames to this
        for i in range( encoder_input.shape[-1] ):
            x = pad_input[:,:,i:i+self._window_size] 
            x = x.transpose(-1,-2)  # our transformer thinks of time as dimension -2, and data as dim -1
            x = self.layers[0](x) # first transformer
            x = x + self.layers[1](x) # secondtransformer (and residual/adding trick)
            x = x.view( (bs, _LATSZ*self._window_size) )
            out.append( self.layers[2](x) )
        # now make the vectors in out into a tensor for output
        res = stack( out, dim=-1 )  # now time is last dimension again
        return res
