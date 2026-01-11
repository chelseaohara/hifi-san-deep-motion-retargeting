'''
organizes, formats, manages data for training
'''
import os
import numpy as np
from data.skeleton_dataset import SkeletonDataset
from data.motion_dataset import MotionDataset
from data.test_dataset import TestDataset
from config.manifest import get_manifest
from utils.bvh_manager import BVHManager

bvh_manager = BVHManager()

class DataManager:
    def __init__(self):
        self.character_datasets = {}

    def __calculate_edge_matrix__(self, edges):
        number_of_edges = len(edges)
        edge_matrix = [[100000] * number_of_edges for _ in range(number_of_edges)]
        for i in range(number_of_edges):
            edge_matrix[i][i] = 0

        for i, edge_a in enumerate(edges):
            for j, edge_b in enumerate(edges):
                link = False
                for x in range(2):
                    for y in range(2):
                        if edge_a[x] == edge_b[y]:
                            link = True
                if link:
                    edge_matrix[i][j] = 1

        for i in range(number_of_edges):
            for j in range(number_of_edges):
                for k in range(number_of_edges):
                    edge_matrix[j][k] = min(edge_matrix[j][k], edge_matrix[j][i] + edge_matrix[i][k])

        return edge_matrix

    def preprocess_data(self):
        '''
        Crawls the bvh.py files for the provided groups and preprocesses the data
        into .npy files for training. Throws exception if any issues arise and exits
        :return: None
        '''
        #try:
        raw_data_dir = get_manifest().DATA.RAW_DIR
        prepared_data_dir = get_manifest().DATA.PREPARED_DIR
        reference_bvh_dir = get_manifest().DATA.REF_BVH_DIR
        mean_var_dir = get_manifest().DATA.MEAN_VAR_DIR

        character_objects = get_manifest().DATA.GROUP_A + get_manifest().DATA.GROUP_B

        for character_object in character_objects:
            character = character_object['character']
            data_path = os.path.join(raw_data_dir, character).replace('\\','/')
            character_bvh_files = sorted([file for file in os.listdir(data_path) if file.endswith('.bvh')])

            if get_manifest().DATA.PREPROCESS:
                # only create .npy files if PREPROCESS flag set to True
                # make a copy of one (arbitrary) bvh file to provide static information about skeleton
                bvh_filepath = '{}/{}/{}'.format(raw_data_dir, character, character_bvh_files[0])
                os.system('cp {} {}_std.bvh'.format(bvh_filepath, reference_bvh_dir+'/'+character))

                print('Concatenating motions from bvh.py files for {}'.format(character))

                motions = []
                for index, bvh_file in enumerate(character_bvh_files):
                    bvh_filepath = '{}/{}/{}'.format(raw_data_dir, character, bvh_file)
                    bvh_file = bvh_manager.load_bvh(bvh_filepath, character_object)
                    if index == 0:
                        print(bvh_file.all_bones)
                        print(bvh_file.base_skeleton_bones)
                    new_motion = bvh_file.to_tensor().permute((1, 0)).numpy()
                    motions.append(new_motion)

                character_motions_filepath = '{}/{}.npy'.format(prepared_data_dir, character)
                motions_array = np.asarray(motions, dtype=object)
                self.save_npy_file(character_motions_filepath, motions_array)

                print('Saving motions of {} to {}'.format(character,prepared_data_dir))

            character_dataset = MotionDataset(character)
            self.character_datasets[character] = character_dataset

            mean_data = character_dataset.mean.cpu().numpy()[0, ...]
            mean_data_filepath = '{}/{}'.format(mean_var_dir, '{}_mean'.format(character))
            self.save_npy_file(mean_data_filepath, mean_data)

            variance_data = character_dataset.var.cpu().numpy()[0, ...]
            variance_data_filepath = '{}/{}'.format(mean_var_dir, '{}_var'.format(character))
            self.save_npy_file(variance_data_filepath, variance_data)

    def create_dataset(self, GROUP_A, GROUP_B):
        '''
        Given reference lists for two groups of characters to use, returns a
        SkeletonDataset object containing the data to use for training.
        :param GROUP_A: string array of character names
        :param GROUP_B: string array of character names
        :return: SkeletonDataset object
        '''
        is_training = get_manifest().TRAINING.IS_TRAINING

        if is_training:
            return SkeletonDataset(GROUP_A, GROUP_B, self.character_datasets)
        else:
            return TestDataset(GROUP_A, GROUP_B, self.character_datasets)

    def save_npy_file(self, save_filepath, numpy_data):
        '''

        :param save_filepath:
        :param numpy_data:
        :return:
        '''
        np.save(save_filepath, numpy_data)

    def build_edge_topology(self, topology, offset):
        '''

        :param topology:
        :param offset:
        :return:
        '''
        edges = []
        number_of_joints = len(topology)
        for i in range(1, number_of_joints):
            edges.append((topology[i], i , offset[i]))
        return edges

    def build_joint_topology(self, edges, origin_names):
        parent = []
        offset = []
        names = []
        edge2joint = []
        joint_from_edge = []  # -1 means virtual joint
        joint_cnt = 0
        out_degree = [0] * (len(edges) + 10)
        for edge in edges:
            out_degree[edge[0]] += 1

        # add root joint
        joint_from_edge.append(-1)
        parent.append(0)
        offset.append(np.array([0, 0, 0]))
        names.append(origin_names[0])
        joint_cnt += 1

        def make_topology(edge_idx, pa):
            nonlocal edges, parent, offset, names, edge2joint, joint_from_edge, joint_cnt
            edge = edges[edge_idx]
            if out_degree[edge[0]] > 1 and out_degree[edge[0]] < 4:
                parent.append(pa)
                offset.append(np.array([0, 0, 0]))
                names.append(origin_names[edge[1]] + '_virtual')
                edge2joint.append(-1)
                pa = joint_cnt
                joint_cnt += 1

            parent.append(pa)
            offset.append(edge[2])
            names.append(origin_names[edge[1]])
            edge2joint.append(edge_idx)
            pa = joint_cnt
            joint_cnt += 1

            for idx, e in enumerate(edges):
                if e[0] == edge[1]:
                    make_topology(idx, pa)

        for idx, e in enumerate(edges):
            if e[0] == 0:
                make_topology(idx, 0)

        return parent, offset, names, edge2joint

    def find_neighbours(self, edges, degree_of_separation):
        edge_matrix = self.__calculate_edge_matrix__(edges)
        neighbours = []
        number_of_edges = len(edge_matrix)

        for i in range(number_of_edges):
            joints_neighbours = []
            for j in range(number_of_edges):
                if edge_matrix[i][j] <= degree_of_separation:
                    joints_neighbours.append(j)
            neighbours.append(joints_neighbours)

        global_part_neighbour = neighbours[0].copy()

        for i in global_part_neighbour:
            neighbours[i].append(number_of_edges)

        neighbours.append(global_part_neighbour)

        return neighbours

    def batch_copy(self, source, suffix, destination, dst_suffix=None):
        if not os.path.exists(source):
            os.makedirs(source, exist_ok=True)
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)

        files = [f for f in os.listdir(source) if f.endswith('_{}.bvh'.format(suffix))]

        length = len('_{}.bvh'.format(suffix))

        for f in files:
            if dst_suffix is not None:
                cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source, f), os.path.join(destination, f[:-length] + '_{}.bvh'.format(dst_suffix)))
            else:
                cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source, f), os.path.join(destination, f[:-length] + '.bvh'))

            os.system(cmd)
