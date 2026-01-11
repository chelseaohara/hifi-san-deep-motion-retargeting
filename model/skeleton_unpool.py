from torch import matmul, nn, Tensor, zeros

class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, channels_per_edge):
        super(SkeletonUnpool, self).__init__()

        self.weight, output_edge = self.__get_setup_params__(pooling_list, channels_per_edge)
        self.description = 'SkeletonUnpool(in_edge_num={}, out_edge_num={})'.format(len(pooling_list), output_edge)

    @staticmethod
    def __get_setup_params__(pooling_list, channels_per_edge):

        output_edge = 0

        for t in pooling_list:
            output_edge += len(t)

        weight = zeros(output_edge * channels_per_edge, len(pooling_list) * channels_per_edge)

        for i, pair in enumerate(pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    weight[j * channels_per_edge + c, i * channels_per_edge + c] = 1

        weight = nn.Parameter(weight)
        weight.requires_grad_(False)

        return weight, output_edge

    def forward(self, unpool_input: Tensor):
        return matmul(self.weight, unpool_input)