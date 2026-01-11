from torch.utils.data import Dataset
from torch import cat, float32, mean, ones_like, tensor, var
import numpy as np

from config.manifest import get_manifest
from utils.quaternions import Quaternions

_MANIFEST = get_manifest()

class MotionDataset(Dataset):
    '''
    MotionDataset is a child class of the PyTorch data primitive torch.utils.data.Dataset, which provides a way to
    store samples and access them with the torch.utils.data.DataLoader primitive.

    A custom Dataset class must implement three functions:
         __init__,
         __len__,
         __getitem__
    Their respective requirements as they pertain to the motion data for Skeleton Motion NN is outlined in each
    function definition.

    Class parameters used in code:
        -
        -
    '''
    def __init__(self, character):
        '''
        Initialization code is run upon MotionData object instantiation. Given the name of a character ...
        :param character:
        '''
        super(MotionDataset, self).__init__()

        # saving the character name for reference, to positively identify this MotionDataset object by character
        self.character_name = character

        # initialize empty array for access to MotionDataset data
        self.data = []

        # initialize length of dataset to zero
        self.length = 0

        # count of frames from each BVH file loaded from /raw for the character
        self.total_frames = 0

        # list of integers representing the sampled frame count for each loaded BVH file
        self.motion_length = []

        file_path = '{}/{}.npy'.format(_MANIFEST.DATA.PREPARED_DIR, character)
        motions = list(np.load(file_path, allow_pickle=True)) #redundant
        new_windows = self.get_windows(motions, get_manifest().DATA.WINDOW_SIZE, get_manifest().DATA.ROTATION_TYPE)

        self.std_bvh = './data/raw/{}.bvh'.format(character)

        self.data.append(new_windows)
        self.data = cat(self.data)
        #self.data = self.data.to(_MANIFEST.SYSTEM.DEVICE)
        self.data = self.data.permute(0,2,1)

        self.mean = mean(self.data, (0, 2), keepdim=True) # this was duplicated in both if/else scenarios below

        if get_manifest().DATA.NORMALIZE:
            # calculate the variance over the dimensions specified by dim
            self.var = var(self.data, (0, 2), keepdim=True)
            # convert to std dev
            self.var = self.var ** (1 / 2)
            idx = self.var < 1e-5 # for each standard deviation in the tensor, assigns False if less than 0.00005, True otherwise
            self.var[idx] = 1 # all entries that are True become 1.0, the rest retain the computed standard deviation value
            self.data = (self.data - self.mean) / self.var
        else:
            # set mean to zero and standard deviation to 1 for all data points
            self.mean.zero_()
            self.var = ones_like(self.mean)

        train_len = self.data.shape[0] * 95 // 100
        self.test_set = self.data[train_len:, ...]
        self.data = self.data[:train_len, ...]
        self.data_reverse = tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = False
        self.virtual_length = 0

    def __len__(self):
        '''
        Returns the length of the MotionData dataset.
        :return: int representing number of motion data frames
        '''
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        '''
        Returns the MotionData object at the provided index
        :param index: int, representing the index
        :return: array, containing column of data from final_data
        '''
        if isinstance(index, int):
            print('Not Implemented -- Determine why the index is changed to %=self.data.shape[0]')
            exit(1)
        if not get_manifest().DATA.AUGMENT:
            return self.data[index]
        else:
            return self.data_reverse[index]

    def get_windows(self, all_motions, window_size, rotation_type):
        '''
        :param all_motions: list of mxn ndarrays, each entry holding the data for 1 BVH file for the character being processed,
                        where m = frames and n = rotation channels for each joint followed by 3 values for position channels
        :param window_size: int
        :param rotation_type: str, should be 'quaternions' or 'euler_angles'
        :return: torch.cat of motion sequences
        '''
        windows = []

        try:
            if window_size % 2 != 0:
                raise Exception("manifest.DATA.WINDOW_SIZE must be even")
        except Exception as e:
            exit(e)

        for single_bvh_motions in all_motions:
            frames = single_bvh_motions.shape[0]
            self.total_frames += frames

            # reduce number of frames by removing every other frame
            # subsample keeps first frame, removes second, keeps third, etc.
            motion_subsample = single_bvh_motions[::2, :]
            subsampled_frames_count = motion_subsample.shape[0]

            # keep the subsample of frames
            self.motion_length.append(subsampled_frames_count)

            step_size = window_size // 2
            number_of_windows = (frames // step_size) - 1

            for i in range(number_of_windows):
                start = i * step_size
                end = start + window_size

                chunk = single_bvh_motions[start:end,:]
                if rotation_type == 'quaternion':
                    # our raw bvh files are in euler by default
                    # pull out rotations for current chunk and convert to Quaternions
                    reshaped_chunk = chunk.reshape(chunk.shape[0],-1,3)
                    extracted_rotations = reshaped_chunk[:,:-1,:]
                    if extracted_rotations.dtype != float:
                        extracted_rotations = extracted_rotations.astype(float)
                    quaternion_rotations = Quaternions.from_euler(np.radians(extracted_rotations)).qs
                    rotations = quaternion_rotations.reshape(quaternion_rotations.shape[0],-1)
                    chunk = np.concatenate((rotations, reshaped_chunk[:,-1,:].reshape(reshaped_chunk.shape[0],-1)), axis=1)

                sliced_chunk = chunk[np.newaxis, ...]
                new_window = tensor(sliced_chunk.astype(float), dtype=float32)
                windows.append(new_window)

        return cat(windows)

    def reset_length(self, length):
        self.reset_length_flag = True
        self.virtual_length = length