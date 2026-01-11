from utils.bvh_loader import BVHLoader
from utils.bvh import BVH
from config.manifest import get_manifest

bvh_loader = BVHLoader

class BVHManager:
    def __init__(self):
        pass

    def get_character_reference_bvh(self, character: str):
        '''
        Given a character name, returns the BVH file path from the reference directory.
        :param character: string - name of character
        :return: path: string - path to BVH file in reference directory
        '''
        ref_dir = get_manifest().DATA.REF_BVH_DIR
        path = '{}/{}_std.bvh'.format(ref_dir, character)
        return path


    def load_bvh(self, path, character_object):
        '''
        creates bvh.py obj
        :param path:
        :param character_object:
        :return:
        '''
        if path is None:
            raise Exception('BVHFile: path not provided to load bvh file')
        skeleton_type = character_object['type']
        character_name = character_object['character']
        # try to load bvh.py file
        bvh_animation, bones_list, delta_t = bvh_loader.load_animation_data(path)
        # create bvh.py object from loaded bvh.py file data
        bvh_obj = BVH(bvh_animation, bones_list, delta_t, skeleton_type, character_name)
        return bvh_obj
