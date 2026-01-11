import sys
sys.path.append('.')
import config
from config.manifest import load_manifest, set_manifest
from utils.evaluate_results import Evaluator
from utils.data_manager import DataManager
import os
import numpy as np

DM = DataManager()

if __name__ == '__main__':
    #ipdb.set_trace()
    # get default values from manifest.py YACS config
    manifest = load_manifest()
    # merge with selected experiment
    manifest.merge_from_file('./config/test_template.yaml')
    manifest.freeze()
    # set merged manifest object as manifest to reference throughout program
    set_manifest(manifest)

    if manifest.SYSTEM.USE_IPDB:
        import ipdb
        ipdb.set_trace()

    test_name = manifest.TEST.TEST_NAME

    print("\n\n","Testing trained model with settings: ", test_name, "\n","-"*54,"\n\n")
    evaluator = Evaluator()

    source_path = os.path.join(manifest.TEST.OUTPUT_DIR, 'bvh')
    intra_destination = os.path.join(manifest.TEST.OUTPUT_DIR, '{}/intra_structure/'.format(test_name))
    cross_destination = os.path.join(manifest.TEST.OUTPUT_DIR, '{}/cross_structure/'.format(test_name))

    intra_error = []
    cross_error = []

    test_characters = manifest.TEST.CHARACTERS

    for i in range(len(test_characters)):
        print('Batch [{}/{}]'.format(i+1, len(test_characters)))

        evaluator.evaluate()

        print('Collecting Error . . .')
        if i == 0:
            # "prefix" from original code is the training directory root './training'
            # ours is './results'
            cross_error += evaluator.get_error(0)
            for character in test_characters:
                character_src_path = os.path.join(source_path, character)
                character_dst_path = os.path.join(cross_destination, character)
                DM.batch_copy(character_src_path, 0, character_dst_path)
                DM.batch_copy(character_src_path, 'gt', character_dst_path, 'gt')

        intra_dst = os.path.join(intra_destination, 'from_{}'.format(test_characters[i]))

        for character in test_characters:
            for character in test_characters:
                character_src_path = os.path.join(source_path, character)
                character_dst_path = os.path.join(intra_dst, character)
                DM.batch_copy(character_src_path, 1, character_dst_path)
                DM.batch_copy(character_src_path, 'gt', character_dst_path, 'gt')

        intra_error += evaluator.get_error(1)

    cross_error = np.array(cross_error)
    intra_error = np.array(intra_error)

    cross_error_mean = cross_error.mean()
    intra_error_mean = intra_error.mean()

    print('now removing bvh folder . . .')
    os.system('rm -r %s' % source_path)

    print('Intra-retargeting error:', intra_error_mean)
    print('Cross-retargeting error:', cross_error_mean)
    print('Evaluation completed.')
