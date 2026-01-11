from torch.utils.data import Dataset
from torch import cat, cuda, device, float, tensor
import numpy as np

from config.manifest import get_manifest
from data.mixed_data import MixedData
from data.motion_dataset import MotionDataset
from utils.bvh_manager import BVHManager

# initialize BVHManager
bvh_manager = BVHManager()

class SkeletonDataset(Dataset):
    '''
    SkeletonDataset is a child class of the PyTorch data primitive torch.utils.data.Dataset, which provides a way to
    store samples and access them with the torch.utils.data.DataLoader primitive. This provides SkeletonDataset the
    functionality of a custom Dataset for the Motion files (bvh.py).

    A custom Dataset class must implement three functions:
         __init__,
         __len__,
         __getitem__
    Their respective requirements as they pertain to the motion data for Skeleton Motion NN is outlined in each
    function definition.

    '''
    def __init__(self, group_a, group_b, character_datasets):
        '''
        Initialization code is run upon SkeletonDataset object instantiation. Given the lists for GroupA characters and
        GroupB characters, assemble the Dataset.
        :param group_a: array of strings, representing characters in group a
        :param group_b: array of strings, representing characters in group b
        :param character_datasets: dict of character
        '''

        self._character_datasets = character_datasets

        # initialize device type
        self._device_type = get_manifest().SYSTEM.DEVICE

        # initialize empty array for access to SkeletonDataset data
        self.final_data = []

        # initialize length of dataset to zero
        self.length = 0

        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.variances = []

        group_a_data, group_a_offsets, group_a_means, group_a_vars = self.__format_group_data__(group_a)
        group_b_data, group_b_offsets, group_b_means, group_b_vars = self.__format_group_data__(group_b)

        all_datas = [group_a_data, group_b_data]

        self.offsets.append(cat(group_a_offsets, dim=0).to(self._device_type))
        self.offsets.append(cat(group_b_offsets, dim=0).to(self._device_type))

        self.means.append(cat(group_a_means, dim=0).to(self._device_type))
        self.means.append(cat(group_b_means, dim=0).to(self._device_type))

        self.variances.append(cat(group_a_vars, dim=0).to(self._device_type))
        self.variances.append(cat(group_b_vars, dim=0).to(self._device_type))

        for dataset_group in all_datas:
            motions = []
            skeleton_index = []
            for index, dataset in enumerate(dataset_group):
                motions.append(dataset[:])
                skeleton_index +=[index]*len(dataset)
            motions_cat = cat(motions, dim=0)
            if self.length != 0 and self.length != len(skeleton_index):
                self.length = min(self.length, len(skeleton_index))
            else:
                self.length = len(skeleton_index)
            self.final_data.append(MixedData(motions_cat, skeleton_index))

    def __format_group_data__(self, group):
        group_data = []
        group_offsets = []
        group_means = []
        group_vars = []

        for index, character in enumerate(group):
            character_name = character['character']
            group_data.append(self._character_datasets[character_name])

            # load mean data from .npy as numpy array and convert to tensor
            mean_data = np.load('{}/{}_mean.npy'.format(get_manifest().DATA.MEAN_VAR_DIR, character_name))
            mean_tensor = tensor(mean_data)
            # reshape tensor and add to means group
            mean = mean_tensor.reshape((1,)+mean_tensor.shape)
            group_means.append(mean)

            # load var data from .npy as numpy array and convert to tensor
            var_data = np.load('{}/{}_var.npy'.format(get_manifest().DATA.MEAN_VAR_DIR, character_name))
            var_tensor = tensor(var_data)
            # reshape tensor and add to vars group
            var = var_tensor.reshape((1,)+var_tensor.shape)
            group_vars.append(var)

            reference_bvh_file_path = '{}/{}_std.bvh'.format(get_manifest().DATA.REF_BVH_DIR, character_name)
            bvh_file = bvh_manager.load_bvh(reference_bvh_file_path, character)

            # create one copy of skeletons and end effectors and add to appropriate list
            if index == 0:
                self.joint_topologies.append(bvh_file.topology)
                self.ee_ids.append(bvh_file.ee_ids)

            offset_data = bvh_file.offsets_by_base_skeleton
            offset_tensor = tensor(offset_data, dtype=float)
            offset = offset_tensor.reshape((1,) + offset_tensor.shape)
            group_offsets.append(offset)

        return group_data, group_offsets, group_means, group_vars

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = []
        for data in self.final_data:
            item.append(data[index])
        return item

    def denormalize(self, gid, pid, data):
        mean = self.means[gid][pid, ...]
        variance = self.variances[gid][pid, ...]
        return data.to(self._device_type) * variance + mean