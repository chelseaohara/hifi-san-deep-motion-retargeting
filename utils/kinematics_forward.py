from math import pi

from sympy.algebras import quaternion
from torch import float, empty, matmul, Tensor, tensor, zeros
from torch import cos as tcos
from torch import sin as tsin
from torch import norm as tnorm

from config.manifest import get_manifest


class ForwardKinematics:
    '''
    Forward Kinematics (FK) solver designed to compute the global positions and orientations of the skeleton structure
    based on local joint rotations and offsets. Supports both Quaternion and Euler angle representations.
    '''
    def __init__(self, edges):
        # NOTE: edges is used to build a list that appears to be identical to the values
        # of the joint_topology tuple

        _MANIFEST = get_manifest()
        self._world = _MANIFEST.DATA.WORLD
        self._order = _MANIFEST.DATA.ORDER
        self._is_quaternion = _MANIFEST.DATA.ROTATION_TYPE == 'quaternion'
        self._position_representation = _MANIFEST.DATA.POSITION_REPRESENTATION

        self._topology, self._rotation_list = self.__setup__(edges)

    def __setup__(self, edges):
        '''
        :param edges: edges is a list of tuples of the form (parent int, child int, offset Tensor)
        :return: topology is a list of indices starting from -1 and indicating the parents
                 rotation_list is a list of indices starting from 1 and increasing by 1 (?)
        '''
        topology = [-1] * (len(edges) + 1)
        rotation_list = []
        for edge in edges:
            # edge[1] is the child/tail bone of that edge
            # assigning value of its parent to its position in the topology index
            topology[edge[1]] = edge[0]
            # append value of child index to list
            rotation_list.append(edge[1])

        return topology, rotation_list

    def _forward(self, rotation: Tensor, position: Tensor, offset: Tensor, world, is_quaternion: bool):
        # it looks like this is only used by forward_from_raw
        if not is_quaternion and rotation.shape[-2] != 3:
            raise Exception('ForwardKinematics: unexpected shape for rotations matrix. For non-quaternion rotations, '
                            'the shape must be 3. The shape is {}'.format(rotation.shape[-2]))

        if is_quaternion and rotation.shape[-2] != 4:
            raise Exception('ForwardKinematics: unexpected shape for rotations matrix. For quaternion rotations, the '
                            'shape must be 4. The shape is {}'.format(rotation.shape[-2]))

        rotation = rotation.permute(0,3,1,2)
        position = position.permute(0,2,1)

        result = empty(rotation.shape[:-1] + (3, ), device = position.device)

        norm = tnorm(rotation, dim=-1, keepdim=True)

        rotation = rotation / norm

        if quaternion:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, self._order)

        offset = offset.reshape((-1,1,offset.shape[-2],offset.shape[-1],1))

        result[...,0,:] = position

        # check that first entry in topology is -1, and raise an exception if not
        if self._topology[0] != -1:
            raise Exception('ForwardKinematics: First entry in Topology is not -1. Topology: {}'.format(self._topology))

        for i, parent_index in enumerate(self._topology):
            if parent_index == -1 and i == 0:
                continue

            transform[..., i,:,:] = matmul(transform[..., parent_index,:,:].clone(),transform[...,i,:,:].clone())
            result[...,i,:] = matmul(transform[..., i,:,:], offset[..., i,:,:]).squeeze()
            if world:
                result[..., parent_index, :] += result[..., parent_index, :]

        return result

    def forward_from_raw(self, raw, offset):
        # NOTE: their implementation of this function has default values for 'world' and 'quaternion'
        # as parameters set to None; however, all instances of this function called in their codebase
        # do not include a parameter value for world or quaternion, and rely on this function's original
        # logic to assign the 'args' value of world and quaternion to the parameters instead. So I have
        # opted to remove these parameters and assign these values based on the MANIFEST called in the
        # init; if it becomes apparent that the world or quaternion flags change for this function in
        # other circumstances, I will reimplement their parameters and logic check

        if self._position_representation == '3D':
            position = raw[:,-3:,:]
            rotation = raw[:,:-3,:]
        elif self._position_representation == '4D':
            raise NotImplementedError('ForwardKinematics: Position representation as 4D is not implemented.')
        else:
            raise Exception('ForwardKinematics: Unknown position representation: {}'.format(self._position_representation))

        if quaternion:
            rotation = rotation.reshape((rotation.shape[0],-1,4,rotation.shape[-1]))
            identity = tensor((1,0,0,0), dtype=float, device = raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0],-1,3,rotation.shape[-1]))
            identity = zeros((3, ), dtype=float, device = raw.device)

        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)

        for i, j in enumerate(self._rotation_list):
            rotation_final[:,j,:,:] = rotation[:,i,:,:]

        return self._forward(rotation=rotation_final, position=position, offset=offset,
                             world=self._world, is_quaternion=self._is_quaternion)

    def from_local_to_world(self, result_to_transform: Tensor):
        result_transformed = result_to_transform.clone()

        for i, parent_index in enumerate(self._topology):
            if parent_index == 0 or parent_index == -1:
                continue
            result_transformed [..., i, :] += result_transformed[..., parent_index, :]

        return result_transformed

    # ---- copy/pasted below from Kinematics.py

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = tcos(euler)
        sin = tsin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

    def transform_from_euler(self, rotation, order):
        rotation = rotation / 180 * pi
        transform = matmul(self.transform_from_axis(rotation[..., 1], order[1]),
                           self.transform_from_axis(rotation[..., 2], order[2]))
        transform = matmul(self.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform