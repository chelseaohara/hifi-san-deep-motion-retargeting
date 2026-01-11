import tqdm
from config.manifest import get_manifest
from model.gan_model import GANModel
from utils.data_manager import DataManager
from utils.bvh_manager import BVHManager
from utils.bvh_loader import BVHLoader
from utils.animation import positions_global

import os
import numpy as np


DM = DataManager()
BVHM = BVHManager()
BVHL = BVHLoader()

_MANIFEST = get_manifest()

class Evaluator:
    def __init__(self):

        self.test_dataset = DM.create_dataset(_MANIFEST.TEST.GROUP_A, _MANIFEST.TEST.GROUP_B)
        self._gan_model = GANModel(self.test_dataset)
        print('LOADING EPOCH {} FROM TRAINED MODELS'.format(_MANIFEST.TEST.NUMBER_OF_EPOCHS))
        self._gan_model.load(epoch=_MANIFEST.TEST.NUMBER_OF_EPOCHS)

    def __batch__(self, suffix, character):
        '''

        '''
        output_path = os.path.join(_MANIFEST.TEST.OUTPUT_DIR, 'bvh')

        all_errors = []

        character_obj = {
            'character': character,
            'type': _MANIFEST.TEST.CHARTYPE
        }
        reference_bvh_file_path = '{}/{}_std.bvh'.format(get_manifest().DATA.REF_BVH_DIR, character)
        bvh_file = BVHM.load_bvh(reference_bvh_file_path, character_obj)
        height = bvh_file.get_height

        test_number = 0

        new_path = os.path.join(output_path, character)
        files = [f for f in os.listdir(new_path) if f.endswith('_{}.bvh'.format(suffix)) and not f.endswith('_gt.bvh') and 'fix' not in f and not f.endswith('_input.bvh')]

        for index, file in enumerate(files):
            filepath = os.path.join(new_path, file)
            print(index, character, filepath)
            animation, bones, framerate = BVHL.load_animation_data_for_testing(filepath)
            test_number += 1
            index = []
            for i, bone in enumerate(bones):
                if 'virtual' in bone:
                    continue
                index.append(i)

            file_ref = filepath[:-6]+'_gt.bvh'
            animation_ref, bones_ref, framerate_ref = BVHL.load_animation_data_for_testing(file_ref)

            positions = positions_global(animation)
            positions_ref = positions_global(animation_ref)

            positions = positions[:, index, :]
            positions_ref = positions_ref[:, index, :]

            error = (positions - positions_ref) * (positions_ref - positions) #squared difference?
            error /= height ** 2
            error = np.mean(error)
            all_errors.append(error)

        all_errors = np.array(all_errors)

        return all_errors.mean()

    def evaluate(self):
        '''

        '''

        for i, motions in enumerate(self.test_dataset):
            self._gan_model.set_input(motions)
            self._gan_model.test()

    def get_error(self, suffix):
        '''

        '''
        results = []
        characters = _MANIFEST.TEST.CHARACTERS

        for character in characters:
            results.append(self.__batch__(suffix, character))

        return results