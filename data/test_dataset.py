from config.manifest import get_manifest
from utils.bvh_manager import BVHManager
from torch.utils.data import Dataset
from torch import cat, float, tensor, zeros_like

import numpy as np
import os
import yaml

_MANIFEST = get_manifest()
bvh_manager = BVHManager()

class TestDataset(Dataset):
    '''
    A PyTorch Dataset comprised of test data and used to perform the model's evaluation
    based on two character domains.

    TestDataset handles the structural preprocessing of BVH-based motion data based on the
    configuration in the Manifest and the skeleton yaml files.
    '''
    def __init__(self, group_a, group_b):
        # use motions.yaml to test with motions from training dataset
        # use motions_unseen.yaml to test with motions the model has not seen before
        with open('config/motions.yaml', 'r') as file:
            motions_file = yaml.load(file, Loader=yaml.SafeLoader)
            self._motion_files = motions_file['FILES']

        for i, motion_file in enumerate(self._motion_files):
            self._motion_files[i] = motion_file.replace(' ', '')

        self.joint_topologies = []
        self.mean = []
        self.var = []
        self.offsets = []
        self.ee_ids = []

        self._all_characters = [group_a, group_b]

        for i, group in enumerate(self._all_characters):
            mean_group = []
            var_group = []
            offsets_group = []

            for j, character in enumerate(group):
                ref_dir = _MANIFEST.DATA.REF_BVH_DIR
                filepath = '{}/{}_std.bvh'.format(ref_dir, character['character'])

                bvh_file = bvh_manager.load_bvh(filepath, character)

                if j == 0:
                    self.joint_topologies.append(bvh_file.topology)
                    self.ee_ids.append(bvh_file.ee_ids)

                new_offset = bvh_file.offsets_by_base_skeleton
                new_offset = tensor(new_offset, dtype=float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)

                mean_data = np.load('{}/{}_mean.npy'.format(_MANIFEST.DATA.MEAN_VAR_DIR, character['character']))
                mean_tensor = tensor(mean_data)
                mean = mean_tensor.reshape((1,) + mean_tensor.shape)
                mean_group.append(mean)

                var_data = np.load('{}/{}_var.npy'.format(_MANIFEST.DATA.MEAN_VAR_DIR, character['character']))
                var_tensor = tensor(var_data)
                var = var_tensor.reshape((1,) + var_tensor.shape)
                var_group.append(var)

            mean_group = cat(mean_group, dim=0).to(_MANIFEST.SYSTEM.DEVICE)
            self.mean.append(mean_group)

            var_group = cat(var_group, dim=0).to(_MANIFEST.SYSTEM.DEVICE)
            self.var.append(var_group)

            offsets_group = cat(offsets_group, dim=0).to(_MANIFEST.SYSTEM.DEVICE)
            self.offsets.append(offsets_group)

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        for i, character_group in enumerate(self._all_characters):
            res_group = []
            ref_shape = None
            for j in range(len(character_group)):
                new_motion = self.get_item(i, j, item)
                if new_motion is not None:
                    new_motion = new_motion.reshape((1,) + new_motion.shape)
                    new_motion = (new_motion - self.mean[i][j]) / self.var[i][j]
                    ref_shape = new_motion
                res_group.append(new_motion)

            if ref_shape is None:
                print('Bad at {}'.format(item))
                return None
            for j in range(len(character_group)):
                if res_group[j] is None:
                    bad_flag = 1
                    res_group[j] = zeros_like(ref_shape)
            if bad_flag:
                print('Bad at {}'.format(item))

            res_group = cat(res_group, dim=0)
            res.append([res_group, list(range(len(character_group)))])
        return res

    def __len__(self):
        return len(self._motion_files)

    def get_item(self, group_index, character_index, motion_file_index):

        character_obj = self._all_characters[group_index][character_index]
        character = character_obj['character']
        if _MANIFEST.TEST.USE_DIRTY:
            src_dir = _MANIFEST.DATA.RAW_DIR_NOISY
        else:
            src_dir = _MANIFEST.DATA.RAW_DIR
        filepath = '{}/{}/{}_{}'.format(src_dir, character, character, self._motion_files[motion_file_index])

        if not os.path.exists(filepath):
            raise Exception('TestDataset: Cannot find file {}'.format(filepath))

        file = bvh_manager.load_bvh(filepath, character_obj)
        motion = file.to_tensor(is_quaternions=_MANIFEST.DATA.ROTATION_TYPE=='quaternion')
        motion = motion[:,::2]

        length = motion.shape[-1]
        length = length // 4 * 4

        return motion[..., :length].to(_MANIFEST.SYSTEM.DEVICE)

    def denormalize(self, gid, pid, data):
        '''
        This function takes in a character data point from the TestDataset that has been normalized (Z-Value) and returns
        its denormalized version. x = z*
        :param gid: group index, 0 or 1 representing the group that the character belongs to (domain A or B) and used to
                    select the appropriate mean and variance values from self.means, self.var
        :param pid: position index, integer representing the position of the character's mean and variance data for
                    selection from self.means, self.var
        :param data:
        '''
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]

        return data * var + means

    def normalize(self, group_id, character_index, data):
        means = self.mean[group_id][character_index, ...]
        var = self.var[group_id][character_index, ...]

        return (data - means) / var