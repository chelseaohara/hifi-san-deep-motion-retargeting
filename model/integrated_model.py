from model.auto_encoder import AutoEncoder
from model.gan_discriminator import Discriminator
from model.static_encoder import StaticEncoder
from config.manifest import get_manifest
from utils.bvh_manager import BVHManager
from utils.data_manager import DataManager
from utils.kinematics_forward import ForwardKinematics

from torch import cat, float, load, save, tensor, zeros
import os


DM = DataManager()
BVHM = BVHManager()
_MANIFEST = get_manifest()

class IntegratedModel:
    '''
    This class provides configurations for the generator and discriminator. It creates an
    AutoEncoder, Discriminator, and StaticEncoder for the group (domain/skeleton type).
    It provides the following public functions:
        - get_generator_parameters()
        - get_discriminator_parameters()
        - save()
        - load()
    and features of the skeleton type it represents:
        - fk -- ForwardKinematics object
        - skeleton_heights -- list of skeleton heights for each character in the group
        - normalized_heights -- modified list of skeleton_heights, format depends on end_effector_loss_fact
        - height_para -- nly populated if the end_effector_loss_fact has a specific value
    '''
    def __init__(self, joint_topology, character_objects):

        if _MANIFEST.TRAINING.OPERATOR != 'simple':
            raise NotImplementedError('IntegratedModel only supports the simple operator')

        self.skeleton_type = character_objects[0]['type']
        # edges are a list of tuples, each tuple contains:
        #     - index of parent (for simplified skeleton topology) -- pa = parent
        #     - order in list starting at 1 -- child =
        #     - tensor with shape (3,) representing 'offsets' -- offsets = [0.,0.,0.]
        # renaming edges to bone_details_list -- only simplified bones included
        bone_details_list = DM.build_edge_topology(joint_topology, zeros((len(joint_topology), 3)))
        self.fk = self.__get_forward_kinematics__(bone_details_list)

        self.skeleton_heights, self.normalized_heights, self.height_para, self.real_height = self.__get_skeleton_heights__(character_objects=character_objects,
                                                              use_separate_end_effectors=_MANIFEST.MODEL.IM_USE_SEPARATE_END_EFFECTORS,
                                                              end_effector_loss_fact=_MANIFEST.MODEL.IM_END_EFFECTOR_LOSS_FACT,
                                                              device = _MANIFEST.SYSTEM.DEVICE)

        # NOTE: for the discriminator and static encoder, the param 'edges' was originally named 'topology' and
        # not sure why they are being used interchangeably, the value being used for all three is self.edges
        self.auto_encoder = AutoEncoder(edges=bone_details_list, skeleton_type=self.skeleton_type).to(_MANIFEST.SYSTEM.DEVICE)
        self.discriminator = Discriminator(edges=bone_details_list, skeleton_type=self.skeleton_type).to(_MANIFEST.SYSTEM.DEVICE)
        self.static_encoder = StaticEncoder(edges=bone_details_list, skeleton_type=self.skeleton_type).to(_MANIFEST.SYSTEM.DEVICE)

    def __get_forward_kinematics__(self, edges):
        return ForwardKinematics(edges)

    def __get_skeleton_heights__(self, character_objects, use_separate_end_effectors: bool, end_effector_loss_fact: str, device: str):
        skeleton_heights = []
        normalized_heights = []
        height_para = []
        real_height = []

        for character in character_objects:
            character_name = character['character']
            character_bvh_path = BVHM.get_character_reference_bvh(character_name)
            bvh = BVHM.load_bvh(character_bvh_path, character)

            real_height.append(bvh.get_height)

            # NOTE: see explanation for get_ee_length() value
            if use_separate_end_effectors:
                print('IntegratedModel: the flag use_separate_end_effectors is set to True. The height will be an array of heights.')
                height = bvh.get_ee_lengths()
            else:
                height = bvh.get_height

            # NOTE: implementing based on one height value for each character, appended to this list
            skeleton_heights.append(height)

            if end_effector_loss_fact == 'learn':
                height_as_tensor = tensor(height, dtype=float)
            else:
                # may be redundant
                height_as_tensor = tensor(height, dtype=float, requires_grad=False)

            normalized_heights.append(height_as_tensor.unsqueeze(0))

        skeleton_heights = tensor(skeleton_heights, device=device)
        normalized_heights = cat(normalized_heights, dim=0).to(device)
        real_height = tensor(real_height, device=device)

        if not use_separate_end_effectors:
            normalized_heights.unsqueeze_(-1)

        if end_effector_loss_fact == 'learn':
            height_para = [normalized_heights]

        return skeleton_heights, normalized_heights, height_para, real_height

    def get_generator_parameters(self):
        auto_encoder_params = list(self.auto_encoder.parameters())
        static_encoder_params = list(self.static_encoder.parameters())
        return auto_encoder_params + static_encoder_params + self.height_para

    def get_discriminator_parameters(self):
        discriminator_params = list(self.discriminator.parameters())
        return discriminator_params

    def parameters(self):
        return self.get_generator_parameters() + self.get_discriminator_parameters()

    def save(self, epoch):
        '''
        This save function is called by the GAN for select epochs. Saves the following .pt files to the appropriate
        directory for this integrated model's skeleton type:
        -   skeleton_heights
        -   auto_encoder
        -   discriminator
        -   static_encoder
        epoch: epoch number ex. 0, 200, etc.
        '''
        save_dir = '{}type{}'.format(_MANIFEST.TRAINING.MODELS_DIR,self.skeleton_type)
        results_path_for_epoch = os.path.join(save_dir, str(epoch))
        if not os.path.exists(results_path_for_epoch):
            os.makedirs(results_path_for_epoch, exist_ok=True)

        save(self.skeleton_heights, os.path.join(results_path_for_epoch, 'height.pt'))
        save(self.auto_encoder.state_dict(), os.path.join(results_path_for_epoch, 'auto_encoder.pt'))
        save(self.discriminator.state_dict(), os.path.join(results_path_for_epoch, 'discriminator.pt'))
        save(self.static_encoder.state_dict(), os.path.join(results_path_for_epoch, 'static_encoder.pt'))

        print('   > successfully saved integrated model {} for epoch {}'.format(self.skeleton_type,epoch))

    def load(self, epoch=None):
        if epoch is None:
            raise Exception('Integrated Model: load() function called with no Epoch')

        load_dir = '{}type{}'.format(_MANIFEST.TRAINING.MODELS_DIR, self.skeleton_type)

        epoch_path = os.path.join(load_dir, str(epoch))

        self.auto_encoder.load_state_dict(load(os.path.join(epoch_path, 'auto_encoder.pt'), map_location=_MANIFEST.SYSTEM.DEVICE))
        self.static_encoder.load_state_dict(load(os.path.join(epoch_path, 'static_encoder.pt'), map_location=_MANIFEST.SYSTEM.DEVICE))
