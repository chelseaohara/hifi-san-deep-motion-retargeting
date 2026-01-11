# Modified sample code from rubenvillegas/cvpr2018nkn
# https://github.com/rubenvillegas/cvpr2018nkn
# I copied this file to the root of the Blender linux instance
# and ran `./blender -b -P ./fbx_converter.py`
# note these values must change:
#     - raw_data_dir
#     - character
#     - motion_file
#     - motion
#     - dump_path
# also added separate path structure for unseen data -- set test to True

import bpy
import numpy as np

test = False

raw_data_dir = '/path/to/data/raw/_fbx'
raw_test_data_dir = '/path/to/data/raw/_fbx/forTesting'
character = 'SportyGranny' # Update
motions_for_testing = ['DancingTwerk', 'Paddling']
motions = ['Acknowledging']

if test:
    target_motions = motions_for_testing
else:
    target_motions = motions

for motion in target_motions:
    motion_file = '{}_{}.fbx'.format(character, motion)
    if test:
        source_path = '{}/{}/{}'.format(raw_test_data_dir, character, motion_file)
        dump_path = '{}/{}_{}.bvh'.format(raw_test_data_dir, character, motion)
    else:
        source_path = '{}/{}/{}'.format(raw_data_dir, character, motion_file)
        dump_path = '{}/{}_{}.bvh'.format(raw_data_dir, character, motion)

    bpy.ops.import_scene.fbx(filepath=source_path)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if action.frame_range[1] > frame_end:
        frame_end = int(action.frame_range[1])
    if action.frame_range[0] < frame_start:
        frame_start = int(action.frame_range[0])

    frame_end = np.max([60, frame_end])
    bpy.ops.export_anim.bvh(filepath=dump_path, frame_start=frame_start, frame_end=frame_end, root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])

print("Completed.")