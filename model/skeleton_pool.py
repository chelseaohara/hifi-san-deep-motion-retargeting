from torch import matmul, nn, Tensor, zeros
from config.manifest import get_manifest
import yaml

class SkeletonPool(nn.Module):
    def __init__(self, edges, channels_per_edge, layer_number, skeleton_type, last_pool):
        '''

        :param REMOVED pooling_mode: string, denoting the type of pooling. Only 'mean' is supported.
        :param edges:
        :param channels_per_edge:
        :param last_pool: REMOVED bool, indicating if this is the final layer
        :param layer_number: int, used as index for pooling list from manifest lookup
        '''
        super(SkeletonPool, self).__init__()
        _MANIFEST = get_manifest()

        self._number_of_edges = len(edges) + 1
        self.pooling_list, self.new_edges = self.__get_pooling__(edges, last_pool)
        self.weight = self.__get_weight__(self.pooling_list, channels_per_edge)

    def __get_pooling__(self, edges, last_pool):
        self.seq_list = []
        pooling_list = []
        new_edges = []
        edge_num = len(edges)+1

        degrees = [0] * 100

        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

        degree = [0] * 100

        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1

        def find_seq(j, seq):
            nonlocal self, degree, edges

            if degree[j] > 2 and j != 0:
                self.seq_list.append(seq)
                seq = []

            if degree[j] == 1:
                self.seq_list.append(seq)
                return

            for idx, edge in enumerate(edges):
                if edge[0] == j:
                    find_seq(edge[1], seq + [idx])

        find_seq(0, [])
        for seq in self.seq_list:
            if last_pool:
                pooling_list.append(seq)
                continue
            if len(seq) % 2 == 1:
                pooling_list.append([seq[0]])
                new_edges.append(edges[seq[0]])
                seq = seq[1:]
            for i in range(0, len(seq), 2):
                pooling_list.append([seq[i], seq[i + 1]])
                new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

        # add global position
        pooling_list.append([edge_num - 1])

        return pooling_list, new_edges

    def __get_weight__(self, pooling_list, channels_per_edge):
        weight = zeros(len(pooling_list) * channels_per_edge, self._number_of_edges * channels_per_edge)
        for i, pair in enumerate(pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    weight[i*channels_per_edge + c, j * channels_per_edge + c] = 1.0 / len(pair)

        weight = nn.Parameter(weight, requires_grad=False)
        return weight

    def forward(self, input: Tensor):
        return matmul(self.weight, input)