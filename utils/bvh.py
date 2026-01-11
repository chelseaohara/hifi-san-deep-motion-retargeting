from utils.quaternions import Quaternions
from config.manifest import get_manifest
from torch import tensor, float, matmul
import numpy as np
import yaml
from utils.kinematics_forward import ForwardKinematics

# rewritten from BVH_file class in bvh_parser.py
class BVH:
    '''
    The bvh.py object provides attributes from a loaded bvh.py file for use by the Data Manager to prepare
    the datasets.
    Members:
    -- all_bones: all bones in the bvh file, ordered by HIERARCHY order as read from bvh file; cleaned to remove any prefix "prefix:"
    -- base_skeleton_bones: shortlist of bones representing the simplified skeleton
    -- base_bone_indices: list of integers representing the index of the bone in bones list
    -- ee_lengths: list of magnitudes representing the total offset for each end effector from the root - Note: NOT SURE HOW TO USE
    '''

    def __init__(self, animation, bones, framerate, skeleton_type, character_name):
        self._character_name = character_name
        # print('INITIALIZING BVH OBJECT FOR {}'.format(character_name))
        self.skeleton_type = skeleton_type
        self.all_bones = self.__get_cleaned_bones__(bones)
        self.animation = self.__check_animation__(animation)
        self.framerate = framerate


        self.base_skeleton_bones, self.base_bone_indices = self.__get_base_skeleton_bones__(
            skeleton_type)  # called self.corps in original code; this is the list of bones of the base skeleton
        self.number_of_base_bones = len(self.base_skeleton_bones)
        self.extra_bones = self.__get_extra_skeleton_bones__()  # list of indices representing the bones that are NOT in the base skeleton i.e. extra bones

        self.number_of_joints = self.animation.shape[1]

        self.simplified_map, self.inverse_simplified_map, self.simplified_names = self.__get_simplified_attributes__()

        self.topology = self.__get_topology__()
        self.edges = self.__get_edges__()
        self.edge_mat = []
        self.edge_num = 0

        self.ee_ids = self.__get_end_effector_ids__(skeleton_type)
        self.__height = self.__get_height__()
        self.__ee_lengths = self.__get_end_effector_lengths__()

    def __check_animation__(self, animation):
        '''
        Encapsulating the type 0 skeleton hack into a function 'check_animation'
        '''
        if self.skeleton_type == 'A' or self.skeleton_type == 'C':
            # below is copied from the set_new_root() function in bvh_parser.py
            # does the following to the Animation obj:
            #   - modifies first entry in parents to be 0 (originally -1, which causes an issue downstream)
            #   - modifies the offsets to use Pelvis (1) as the root (instead of Hips (0))
            #   - relabels bones based on sequence created from new root
            #   - updates self.all_bones list based on new bone sequence
            new_root =1
            euler = tensor(animation.rotations[:, 0, :], dtype=float)
            transform = ForwardKinematics.transform_from_euler(ForwardKinematics, euler, 'xyz')
            offset = tensor(animation.offsets[new_root], dtype=float)
            new_pos = matmul(transform, offset)
            new_pos = new_pos.numpy() + animation.positions[:, 0, :]
            animation.offsets[0] = -animation.offsets[new_root]
            animation.offsets[new_root] = np.zeros((3,))
            animation.positions[:, new_root, :] = new_pos
            rot0 = Quaternions.from_euler(np.radians(animation.rotations[:, 0, :]), order='xyz')
            rot1 = Quaternions.from_euler(np.radians(animation.rotations[:, new_root, :]), order='xyz')
            new_rot1 = rot0 * rot1
            new_rot0 = (-rot1)
            new_rot0 = np.degrees(new_rot0.euler())
            new_rot1 = np.degrees(new_rot1.euler())
            animation.rotations[:, 0, :] = new_rot0
            animation.rotations[:, new_root, :] = new_rot1

            new_seq = []
            vis = [0] * animation.rotations.shape[1]
            new_idx = [-1] * len(vis)
            new_parent = [0] * len(vis)

            def relabel(x):
                nonlocal new_seq, vis, new_idx, new_parent
                new_idx[x] = len(new_seq)
                new_seq.append(x)
                vis[x] = 1
                for y in range(len(vis)):
                    if not vis[y] and (animation.parents[x] == y or animation.parents[y] == x):
                        relabel(y)
                        new_parent[new_idx[y]] = new_idx[x]

            relabel(new_root)
            animation.rotations = animation.rotations[:, new_seq, :]
            animation.offsets = animation.offsets[new_seq]
            names = self.all_bones.copy()
            for i, j in enumerate(new_seq):
                self.all_bones[i] = names[j]
            animation.parents = np.array(new_parent, dtype=np.int_)

        return animation
    def __get_cleaned_bones__(self, bones):
        '''
        Private. Cleans bone names to remove character name if present.
        :param bones: raw list of bones as read from bvh.py file
        :return: array - list of bone names without character name
        '''
        for index, bone_name in enumerate(bones):
            if ':' in bone_name:
                stripped_bone_name = bone_name.split(':')[1]
                bones[index] = stripped_bone_name
        return bones

    def __get_base_skeleton_bones__(self, skeleton_type):
        '''
        Private. Returns lists of bones/integers for corresponding skeleton type.
        :param skeleton_type: string character ex. 'A' representing the skeleton type lookup
        :return:
        '''
        bone_indices = []
        with open('config/skeletons.yaml', 'r') as file:
            skeleton_data = yaml.load(file, Loader=yaml.SafeLoader)
            base_bones = skeleton_data['TYPE'][skeleton_type]['BONES']
        file.close()
        for base_bone_name in base_bones:
            if base_bone_name not in self.all_bones:
                print('Missing bone {}'.format(base_bone_name))
            for index, bone_name in enumerate(self.all_bones):
                if base_bone_name == bone_name:
                    bone_indices.append(index)
        new_indices = []
        for bone_name in base_bones:
            for index in range(len(self.all_bones)):
                if bone_name == self.all_bones[index]:
                    new_indices.append(index)
                    break
        return base_bones, bone_indices

    def __get_extra_skeleton_bones__(self):
        '''
        Private. Returns array of integers representing indices of bones in full bones list that are
        not in the base skeleton bones list i.e. the 'extra' bones
        :return: array - integers representing indices of bones in self.all_bones
        '''
        extra_bones = [i for i, name in enumerate(self.all_bones) if name not in self.base_skeleton_bones]
        return extra_bones

    def __get_topology__(self):
        topology = self.animation.parents[self.base_bone_indices].copy()
        for i in range(topology.shape[0]):
            if i >= 1:
                topology[i] = self.simplified_map[topology[i]]
        return tuple(topology)

    def __get_edges__(self):
        edges = []
        number_of_joints = len(self.topology)
        animation_offsets = self.animation.offsets[self.base_bone_indices]
        for i in range(1, number_of_joints):
            edges.append((self.topology[i], i, animation_offsets[i]))
        return edges

    def __get_end_effector_ids__(self, skeleton_type):
        end_effector_indices = []
        with open('config/skeletons.yaml', 'r') as file:
            skeleton_data = yaml.load(file, Loader=yaml.SafeLoader)
            end_effectors = skeleton_data['TYPE'][skeleton_type]['END_EFFECTORS']
            for end_effector in end_effectors:
                end_effector_indices.append(self.base_skeleton_bones.index(end_effector))
        return end_effector_indices

    def __get_end_effector_lengths__(self):
        ee_lengths = []

        # configure a list 'degree' with entries corresponding to each integer in topology
        degree = [0] * len(self.topology)

        for bone_index in self.topology:
            if bone_index > -1:
                # if entry has a parent, update degree list count for that bone's parent
                degree[bone_index] += 1
        # degree list represents a count of how many children that bone has

        for index, ee_index in enumerate(self.ee_ids):
            # we will traverse the topology list from an end effector and work backwards towards
            # the previous end effector in ee_index list
            length = 0
            # define the 'stop' for our range (start is the current end effector index)
            if index == 0:
                stop = 0
            else:
                stop = self.ee_ids[index - 1]
            # starting with end effector, sum magnitudes of offset vectors (i.e. add to length)
            for i in range(ee_index, stop, -1):
                bone_offset = self.offsets_by_base_skeleton[i]
                length += np.dot(bone_offset, bone_offset) ** 0.5
            ee_lengths.append(length)

        end_effectors_grouped = [[0,1],[2],[3,4]]
        for group in end_effectors_grouped:
            if len(group) > 1:
                larger = max(ee_lengths[group[0]], ee_lengths[group[1]])
            else:
                larger = ee_lengths[group[0]]
            for bone in group:
                ee_lengths[bone] *= self.__height/larger

        return ee_lengths

    def __get_simplified_attributes__(self):
        simplified_map = {}
        inverse_simplified_map = {}
        simplified_names = []
        for index, bone in enumerate(self.base_bone_indices):
            simplified_map[bone] = index
            inverse_simplified_map[index] = bone
            simplified_names.append(self.all_bones[index])
        inverse_simplified_map[0] = -1
        for i in range(self.animation.shape[1]):
            if i in self.extra_bones:
                simplified_map[i] = -1
        return simplified_map, inverse_simplified_map, simplified_names

    def __get_height__(self):
        summed_magnitudes = 0 # I think this is magnitude? (offset = 1x3 matrix dot itself) ** 0.5

        # first end effector in list, end effectors are indices representing bones from the
        # simplified skeleton
        selected_end_effector = self.ee_ids[0] # index of parent

        # start from end effector identified in parent list and iterate down to 0 (root)
        for i in range(selected_end_effector, 0, -1):
            if self.topology[i] == 0:
                break
            summed_magnitudes += np.dot(self.offsets_by_base_skeleton[i], self.offsets_by_base_skeleton[i])** 0.5

        # repeat for 'head'
        selected_end_effector = self.ee_ids[2]

        for i in range(selected_end_effector, 0, -1):
            if self.topology[i] == 0:
                break
            summed_magnitudes += np.dot(self.offsets_by_base_skeleton[i], self.offsets_by_base_skeleton[i]) ** 0.5

        return summed_magnitudes

    @property
    def offsets_by_base_skeleton(self):
        return self.animation.offsets[self.base_bone_indices]

    @property
    def get_ee_lengths(self):
        return self.__ee_lengths

    @property
    def get_height(self):
        return self.__height

    def to_numpy(self, is_quaternions, is_edge):
        rotations = self.animation.rotations[:, self.base_bone_indices, :]
        positions = self.animation.positions[:, 0, :]

        if is_quaternions:
            rotations = Quaternions.from_euler(np.radians(rotations)).qs

        if is_edge:
            edges_by_indices = []
            for e in self.edges:
                edges_by_indices.append(e[0])
            rotations = rotations[:, edges_by_indices, :]

        reshaped_rotations = rotations.reshape(rotations.shape[0], -1)
        return np.concatenate((reshaped_rotations, positions), axis=1)

    def to_tensor(self, is_quaternions=False, is_edge=True):
        rotations_positions_array = self.to_numpy(is_quaternions, is_edge)
        torch_tensor_array = tensor(rotations_positions_array, dtype=float)
        flipped_torch_tensor_array = torch_tensor_array.permute(1, 0)
        final_tensor = flipped_torch_tensor_array.reshape((-1, flipped_torch_tensor_array.shape[-1]))
        return final_tensor

