from torch.utils.data import Dataset
from torch import rand, tensor
from config.manifest import get_manifest

class MixedData(Dataset):
    '''
    MixedData is a child class of the PyTorch data primitive torch.utils.data.Dataset. It is used by the
    SkeletonDataset in its construction of the final dataset.
    '''
    def __init__(self, motions, skeleton_index):
        super(MixedData, self).__init__()

        self.motions = motions
        self.motions_reversed = tensor(self.motions.numpy()[..., ::-1].copy())

        self.skeleton_index = skeleton_index

        self.length = motions.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if get_manifest().DATA.AUGMENT == False or rand(1) < 0.5:
            return [self.motions[index], self.skeleton_index[index]]
        else:
            return [self.motions_reversed[index], self.skeleton_index[index]]
