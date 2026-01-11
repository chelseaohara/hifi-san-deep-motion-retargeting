# Adapted from:
#   Holden et al.
#   Aberman et al.
# and heavily commented by me - C. O'Hara
from utils.quaternions import Quaternions
import numpy as np
import re

from utils.animation import Animation

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'
}

class BVHLoader:
    rotations_channel_map = {
        'Xrotation': 'x',
        'Yrotation': 'y',
        'Zrotation': 'z'
    }

    def __init__(self):
        pass

    def __get_axis__(self, channel_name):
        return self.rotations_channel_map[channel_name]

    @classmethod
    def load_animation_data(cls, filename, start=0, end=None, order=None, world=False, need_quaternion=False):
        '''
        Reads a bvh.py file and constructs an Animation object.
        :param filename:
        :param start: int - index of first frame. Defaults to 0
        :param end:
        :param order: string. Represents order of x y z axis example 'xyz'. Defaults to None. If None, set by parsing process
        :param world:
        :param need_quaternion:
        :return: tuple. Three values 1. Animation obj 2. joint_names 3. framerate
        '''
        open_file = open(filename, 'r')

        # numeric indicator for number of frames hit by frame counter if start and end values provided
        frame_count = 0

        # numeric indicator for level of bone hierarchy from root (-1)
        # increases as hierarchy is traversed and bone data is extracted from bvh
        current_position = -1
        end_site = False

        animation = None
        bones = []
        channels_by_bone = []
        number_of_frames = 0
        framerate = 0

        orientations = Quaternions.id(0)
        offsets = np.array([]).reshape((0,3))
        parents = np.array([],dtype=int)
        rotations = []
        positions = []

        for line in open_file:
            # --- instructions for headers and brackets and end effectors
            if "HIERARCHY" in line or "MOTION" in line or "{" in line:
                # skip header lines and go to next line
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    current_position = parents[current_position]
                continue

            if "End Site" in line:
                end_site = True
                continue

            # --- instructions for root bone
            # Note: some of the mixamo data has a namespace attached to the root bone name separated by a
            # colon, i.e. name:name, and there is a second regex pattern used to find these
            #           (\w+)       finds single alphanumeric word after "ROOT "
            #           (\w+:?\w+)  finds alphanumeric words separated by : after "ROOT "
            is_root_match = re.match(r"ROOT (\w+:?\w+)", line)
            if is_root_match:
                # is_root_match.group(0) is the full line "ROOT {bone}"
                # is_root_match.group(1) is the "{bone}" name
                bones.append(is_root_match.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]]), axis=0)
                orientations.qs = np.append(orientations.qs, np.array([[1,0,0,0]]), axis=0)
                parents = np.append(parents, current_position)
                current_position = len(parents)-1
                # found the root so go to next line
                continue

            # --- instructions for bone offsets
            is_offset_match = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if is_offset_match:
                if not end_site:
                    offsets_array = [list(map(float, is_offset_match.groups()))]
                    offsets[current_position] = np.array(offsets_array)
                # found offsets information so go to next line
                continue

            # --- instructions for bone channels
            is_channel_match = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if is_channel_match:
                number_of_channels = int(is_channel_match.group(1))
                # record channel count
                channels_by_bone.append(number_of_channels)
                if order is None: # if order has not been set by bvh.py obj creation, must read from bvh.py file

                    # based on the number of channels stated at the beginning of the line, set some offset amounts for
                    # properly selecting the 'rotations' channel names
                    if number_of_channels == 3:
                        start_index_offset = 0
                        end_index_offset = 3
                    elif number_of_channels == 6:
                        start_index_offset = 3
                        end_index_offset = 6
                    else:
                        raise NotImplementedError('Detected {} channels while reading bvh.py file.'.format(number_of_channels))
                    # create list of rotation channel names
                    channels_list = line.split()[2+start_index_offset:2+end_index_offset]
                    for channel in channels_list:
                        if channel not in cls.rotations_channel_map:
                            continue # apparently ok to do this and not throw NotImplemented to make it 'less strict' on messy motion data
                    # kind of long but more explicit that we have 3 axis and order matters
                    order = "{}{}{}".format(cls.__get_axis__(cls, channels_list[0]), cls.__get_axis__(cls, channels_list[1]), cls.__get_axis__(cls, channels_list[2]))
                    # order should get set by first CHANNELS and not again....

                # only used to set order? only done once?
                continue

            # --- instructions for joint
            is_joint_match = re.match(r"\s*JOINT\s+(\w+:?\w+)", line)
            if is_joint_match:
                bones.append(is_joint_match.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]]), axis=0)
                orientations.qs = np.append(orientations.qs, np.array([[1,0,0,0]]), axis=0)
                parents = np.append(parents, current_position)
                current_position = len(parents)-1
                continue

            # --- instructions for number of frames
            # Note: once frames-related line is hit, the joint hierarchy has been traversed
            is_frame_number = re.match("\s*Frames:\s+(\d+)", line)
            if is_frame_number:
                if start and end:
                    number_of_frames = (end-start)-1
                else:
                    number_of_frames = int(is_frame_number.group(1))
                number_of_joints = len(parents) # this is also not used in original code
                # NOTE: this is where positions and rotations arrays are initially populated
                #    positions: (number of frames x (number of bones x offsets))
                #    rotations: (number of frames x (number of bones x rotations))
                positions = offsets[np.newaxis].repeat(number_of_frames, axis=0)
                rotations = np.zeros((number_of_frames, len(orientations),3))
                continue

            # --- instructions for frame rate
            is_framerate = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if is_framerate:
                framerate = float(is_framerate.group(1))
                continue

            # --- logic for frame counter
            # if start and end values provided, we need to count frames to know which ones to extract data for
            if end and (frame_count < start or frame_count >= end - 1):
                # this was never actually used in the original codebase so I will refrain from implementing for now
                frame_count += 1
                continue

            # --- logic for parsing motion data from line if line is an array of frame data for the skeleton
            # (if this code is hit in the loop, any remaining lines must be motions)
            frame_channel_data = line.strip().split()
            frame_data_array = np.array(list(map(float, frame_channel_data)))
            number_of_bones = len(parents)
            frame_index = frame_count - start
            root_bone_channels = channels_by_bone[0]
            child_bone_channels = channels_by_bone[1]

            if child_bone_channels == 3:
                positions[frame_index, 0:1] = frame_data_array[0:3]
                rotations[frame_index, : ] = frame_data_array[3: ].reshape(number_of_bones,3)
            else:
                raise NotImplementedError('This code assumes bvh.py files with 3 channels for children of root bones.')

            frame_count += 1

        open_file.close()

        if need_quaternion:
            # Note: in the original code, this is always false and so these rotations are never converted
            rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)

        if order != 'xyz':
            # Note: the bvh.py files used are likely to be xyz, so this also would not be hit
            rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
            rotations = np.degrees(rotations.euler())

        animation = Animation(rotations, positions, orientations, offsets, parents)
        return animation, bones, framerate

    @classmethod
    def load_animation_data_for_testing(cls, filename, start=None, end=None, order=None, world=False):
        """
        Reads a BVH file and constructs an animation

        Parameters
        ----------
        filename: str
            File to be opened

        start : int
            Optional Starting Frame

        end : int
            Optional Ending Frame

        order : str
            Optional Specifier for joint order.
            Given as string E.G 'xyz', 'zxy'

        world : bool
            If set to true euler angles are applied
            together in world space rather than local
            space

        Returns
        -------

        (animation, joint_names, frametime)
            Tuple of loaded animation and joint names
        """

        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False

        names = []
        orients = Quaternions.id(0)
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in f:

            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                if start and end:
                    fnum = (end - start) - 1
                else:
                    fnum = int(fmatch.group(1))
                jnum = len(parents)
                # result: [fnum, J, 3]
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                # result: [fnum, len(orients), 3]
                rotations = np.zeros((fnum, len(orients), 3))
                continue

            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue

            if (start and end) and (i < start or i >= end - 1):
                i += 1
                continue

            dmatch = line.strip().split()
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    # This should be root positions[0:1] & all rotations
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    # fill in all positions
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)

        return (Animation(rotations, positions, orients, offsets, parents), names, frametime)
