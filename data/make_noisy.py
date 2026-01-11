import os
import numpy as np
from config.manifest import load_manifest, set_manifest

# Standalone script for noising data
# Can add White Gaussian Noise and/or Random Zeroes as required

def make_noisy(src_file, dst_file, noise=0.0, dropout_probability=0.0):
    with open(src_file, 'r') as f:
        lines = f.readlines()

    motion_header_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('MOTION'):
            motion_header_index = i
            break

    lines_pre_motion = lines[:motion_header_index+3]
    rotations_lines = lines[motion_header_index+3:]
    raw_data = [list(map(float, line.strip().split())) for line in rotations_lines if line.strip()]
    data_matrix = np.array(raw_data)
    rotations = data_matrix[:, 3:]

    if noise > 0:
        jitter = np.random.normal(0,noise,rotations.shape)
        rotations += jitter

    if dropout_probability > 0:
        mask = np.random.rand(*rotations.shape) < dropout_probability
        rotations[mask] = 0.0

    data_matrix[:, 3:] = rotations

    with open(dst_file, 'w') as f:
        f.writelines(lines_pre_motion)
        for row in data_matrix:
            line = " ".join([f"{x:.6f}" for x in row])
            f.write(line + "\n")

def generate_noisy_data(characters):
    raw_data = manifest.DATA.RAW_DIR
    output_dir = manifest.data.RAW_DIR_NOISY

    noise = manifest.DATA.NOISE
    dropout_probability = manifest.DATA.DROPOUT_PROBABILITY

    for character in characters:
        character_dir = '{}/{}'.format(raw_data, character)
        character_output_dir = '{}/{}'.format(output_dir, character)

        if not os.path.exists(character_output_dir):
            os.makedirs(character_output_dir)

        character_files = [f for f in os.listdir(character_dir) if f.endswith('.bvh')]

        for file in character_files:
            src_file = os.path.join(character_dir, file)
            dst_file = os.path.join(character_output_dir, file)
            make_noisy(src_file, dst_file, noise, dropout_probability)

if __name__ == '__main__':
    print('Making noisy data . . .')
    manifest = load_manifest()
    manifest.freeze()
    set_manifest(manifest)
    # replace character names for directory names of characters to process below
    characters = [
        'CharacterName01',
        'CharacterName02'
        # etc..
    ]
    generate_noisy_data(characters)