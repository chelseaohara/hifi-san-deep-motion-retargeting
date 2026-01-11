from config.manifest import get_manifest
from model.base_model import BaseModel
from model.integrated_model import IntegratedModel
from model.gan_loss import GANLoss
from model.gan_criterion import Criterion_EE
from model.image_pool import ImagePool
from model.gan_evaluate_criterion import EvaluateCriterion
from utils.bvh_manager import BVHManager
from utils.bvh_writer import BVHWriter

from torch import load, optim, randint, save, tensor
import torch.nn as nn
import os

_MANIFEST = get_manifest()

class GANModel(BaseModel):
    '''
    This model architecture learns the translation of skeletal motion between two distinct
    topologies (Group A and Group B) as defined in the Manifest.
    '''
    def __init__(self, dataset):
        super(GANModel, self).__init__()
        # load manifest for training parameters

        self.dataset = dataset

        # self.n_topology in original code -- their architecture supports 2 groups only
        self._number_of_groups = _MANIFEST.DATA.NUMBER_OF_GROUPS

        # this seems to be set by the set_input function which is called in a loop
        # in train.py of the original code
        # looks like it is used internally so setting to protected
        self._motions = None
        self._motions_backup = None

        self.characters = []
        self.character_objects = _MANIFEST.DATA.GROUP_A + _MANIFEST.DATA.GROUP_B
        for character in self.character_objects:
            self.characters.append(character['character'])

        if _MANIFEST.TRAINING.IS_TRAINING:
            self.characters_by_group = [_MANIFEST.DATA.GROUP_A, _MANIFEST.DATA.GROUP_B]
        else:
            # characters are empty from DATA yaml, need to populate with TEST yaml
            self.characters_by_group = [_MANIFEST.TEST.GROUP_A, _MANIFEST.TEST.GROUP_B]
            self.character_objects = _MANIFEST.TEST.GROUP_A + _MANIFEST.TEST.GROUP_B
            for character in self.character_objects:
                self.characters.append(character['character'])
        # NOTE: these are just called `models` in the original code
        self.nn_models = []

        self.generator_parameters = []
        self.discriminator_parameters = []

        self.discriminator_loss = None

        self.__setup_nn__()

        # there is further instructions for initialization to implement based on
        # whether the IS_TRAINING flag is set to True or not

        if _MANIFEST.TRAINING.IS_TRAINING:
            self._generator_optimizer = optim.Adam(self.generator_parameters, _MANIFEST.MODEL.LEARNING_RATE, betas=(0.9, 0.999))
            self._discriminator_optimizer = optim.Adam(self.discriminator_parameters, _MANIFEST.MODEL.LEARNING_RATE, betas=(0.9, 0.999))
            self.optimizers = [self._generator_optimizer, self._discriminator_optimizer]
            # configure criteria for back prop
            self.criterion_rec = nn.MSELoss()
            self.criterion_gan = GANLoss(_MANIFEST.MODEL.GAN_MODE)
            self.criterion_cycle = nn.L1Loss()
            self.criterion_ee = Criterion_EE(nn.MSELoss())
            # configure fake pools
            self.fake_pools = []
            for i in range(self._number_of_groups):
                self.fake_pools.append(ImagePool(_MANIFEST.MODEL.POOL_SIZE)) # image pool is CycleGAN holdover
        else:
            BVHM = BVHManager()

            self.error_criterion = []
            for i in range(self._number_of_groups):
                self.error_criterion.append(EvaluateCriterion(dataset.joint_topologies[i]))
            self.id_test = 0
            self.bvh_path = os.path.join(_MANIFEST.TEST.OUTPUT_DIR, 'bvh')
            if not os.path.exists(self.bvh_path):
                os.makedirs(self.bvh_path,exist_ok=True)

            self.writer = []

            for i in range (self._number_of_groups):
                writer_group = []
                for character in self.characters_by_group[i]:
                    std_bvh_path = os.path.join(_MANIFEST.DATA.REF_BVH_DIR, '{}_std.bvh'.format(character['character']))
                    file = BVHM.load_bvh(std_bvh_path, character)
                    writer_group.append(BVHWriter(file.edges, file.base_skeleton_bones))
                self.writer.append(writer_group)

    def __setup_nn__(self):
        '''
        This function configures the core settings for the GANModel object before the training/other model activities
        are executed. Loops over the number of groups (domains/skeleton types) provided by the manifest, and for each
        group, creates an integrated_model that is used to set discriminator and generator parameters
        '''
        for i in range(self._number_of_groups):
            integrated_model = IntegratedModel(self.dataset.joint_topologies[i], self.characters_by_group[i])
            self.nn_models.append(integrated_model)
            self.discriminator_parameters += integrated_model.get_discriminator_parameters()
            self.generator_parameters += integrated_model.get_generator_parameters()

    def _get_end_effector(self, position, joint_topology, end_effector_ids, velo=False, from_root=False):
        # originally called get_ee()
        position_copy = position.clone()

        for i, fa in enumerate(joint_topology):
            if i == 0:
                continue
            if not from_root and fa == 0:
                continue
            position_copy[:, :, i, :] += position_copy[:, :, fa, :]

        position_copy = position_copy[:, :, end_effector_ids, :]

        if velo:
            position_copy = position_copy[:, 1:, ...] - position_copy[:, :-1, ...]
            position_copy = position_copy * 10

        return position_copy

    def _forward(self):
        '''
        called by self.optimize_parameters()
        :return:
        '''
        is_training = _MANIFEST.TRAINING.IS_TRAINING
        device = _MANIFEST.SYSTEM.DEVICE
        number_of_layers = _MANIFEST.MODEL.NUMBER_OF_LAYERS
        ee_velo = _MANIFEST.DATA.EE_VELO
        ee_from_root = _MANIFEST.DATA.EE_FROM_ROOT

        character_offsets = [] # self.offset_repr in og code

        for i in range(self._number_of_groups):
            character_offsets.append(self.nn_models[i].static_encoder(self.dataset.offsets[i]))

        self.motions = [] # this is distinct from self._motions which is set for each epoch iteration
        self.denormalized_motions = [] # self.motion_denorm in og code
        self.latents = [] # self.latents in og code
        self.results = [] # self.res in og code
        self.results_positions = [] # self.res_pos in og code
        self.results_denormalized = [] # self.res_denorm in og code

        self.positions_reference = []
        self.end_effectors_reference = []


        for i in range(self._number_of_groups):
            motion, offset_index = self._motions[i]
            prepped_motion = motion.to(device)
            self.motions.append(prepped_motion)

            motion_denorm = self.dataset.denormalize(gid=i, pid=offset_index, data=motion)
            self.denormalized_motions.append(motion_denorm)

            # pull the latent offsets for the current group
            offsets = [character_offsets[i][p][offset_index] for p in range(number_of_layers + 1)]

            # auto_encoder is AE from enc_and_dec, res is reconstruction of motion, latent is latent version
            # (and I think that motion and reconstruction units are angles) -RG
            latent, result = self.nn_models[i].auto_encoder(motion, offsets)

            self.latents.append(latent)
            self.results.append(result)

            result_denormalized = self.dataset.denormalize(gid=i, pid=offset_index, data=result)
            self.results_denormalized.append(result_denormalized)

            result_position = self.nn_models[i].fk.forward_from_raw(result_denormalized, self.dataset.offsets[i][offset_index])
            self.results_positions.append(result_position)

            position = self.nn_models[i].fk.forward_from_raw(motion_denorm, self.dataset.offsets[i][offset_index]).detach()
            self.positions_reference.append(position)

            end_effector = self._get_end_effector(position, self.dataset.joint_topologies[i],
                                                  self.dataset.ee_ids[i], ee_velo, ee_from_root)
            height = self.nn_models[i].normalized_heights[offset_index]
            reshaped_height = height.reshape((height.shape[0], 1, height.shape[1], 1))
            end_effector /= reshaped_height
            self.end_effectors_reference.append(end_effector)

        random_indexes = []
        fake_results = []
        self.fake_results_denormalized = []
        self.fake_latents = []
        self.fake_positions = []
        self.fake_end_effectors = []


        for source in range(self._number_of_groups):
            for target in range(self._number_of_groups):
                if is_training:
                    random_index = randint(len(self.character_objects[target]), (self.latents[source].shape[0],))
                else:
                    random_index = list(range(self.latents[0].shape[0]))
                random_indexes.append(random_index)

                target_offsets = [character_offsets[target][_][random_index] for _ in range(number_of_layers + 1)]
                fake_result = self.nn_models[target].auto_encoder.decoder(self.latents[source], target_offsets)
                fake_results.append(fake_result)

                fake_latent = self.nn_models[target].auto_encoder.encoder(fake_result, target_offsets)
                self.fake_latents.append(fake_latent)

                fake_result_denormalized = self.dataset.denormalize(target, random_index, fake_result)
                self.fake_results_denormalized.append(fake_result_denormalized)

                fake_position = self.nn_models[target].fk.forward_from_raw(fake_result_denormalized, self.dataset.offsets[target][random_index])
                self.fake_positions.append(fake_position)

                fake_end_effector = self._get_end_effector(fake_position, self.dataset.joint_topologies[target],
                                                           self.dataset.ee_ids[target], ee_velo, ee_from_root)
                height = self.nn_models[target].normalized_heights[random_index]
                reshaped_height = height.reshape((height.shape[0], 1, height.shape[1], 1))
                fake_end_effector = fake_end_effector / reshaped_height

                self.fake_end_effectors.append(fake_end_effector)

    def _set_discriminator_grad_requirement(self, requires_grad: bool):
        for model in self.nn_models:
            for parameter in model.discriminator.parameters():
                parameter.requires_grad = requires_grad

    def _backward_generator(self):
        self.recorded_losses = []
        recorded_loss = 0
        self.cycle_loss = 0
        self.generator_loss = 0
        self.ee_loss = 0
        self.total_generator_loss = 0

        for i in range(self._number_of_groups):
            rec_loss1 = self.criterion_rec(self.motions[i], self.results[i])
            self.loss_recorder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            height = self.nn_models[i].real_height[self._motions[i][1]]
            height = height.reshape(height.shape + (1,1,))
            input_position = self.denormalized_motions[i][:,-3:,:]/height
            recorded_position = self.results_denormalized[i][:,-3:,:]/height
            rec_loss2 = self.criterion_rec(input_position, recorded_position)
            self.loss_recorder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            position_ref_global = self.nn_models[i].fk.from_local_to_world(self.positions_reference[i])/height.reshape(height.shape + (1,))
            position_result_global = self.nn_models[i].fk.from_local_to_world(self.results_positions[i])/height.reshape(height.shape + (1,))
            rec_loss3 = self.criterion_rec(position_ref_global, position_result_global)
            self.loss_recorder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)

            loss = rec_loss1 + ((rec_loss2 * _MANIFEST.TRAINING.LAMBDA_GLOBAL_POSE) +
                                (rec_loss3 * _MANIFEST.TRAINING.LAMBDA_POSITION))*100

            self.recorded_losses.append(loss)
            recorded_loss += loss

        p = 0
        for src in range(self._number_of_groups):
            for dst in range(self._number_of_groups):
                cycle_loss = self.criterion_cycle(self.latents[src], self.fake_latents[p])
                self.loss_recorder.add_scalar('cycle_loss_{}_{}'.format(src,dst), cycle_loss)
                self.cycle_loss += cycle_loss

                ee_loss = self.criterion_ee(self.end_effectors_reference[src], self.fake_end_effectors[p])
                self.loss_recorder.add_scalar('ee_loss_{}_{}'.format(src,dst), ee_loss)
                self.ee_loss += ee_loss

                if src != dst:
                    if _MANIFEST.MODEL.GAN_MODE !='default':
                        g_loss = self.criterion_gan(self.nn_models[dst].discriminator(self.fake_positions[p]), True)
                    else:
                        g_loss = tensor(0)
                    self.loss_recorder.add_scalar('generator_loss_{}_{}'.format(src,dst), g_loss)
                    self.generator_loss += g_loss

                p += 1

        self.total_generator_loss = recorded_loss * _MANIFEST.TRAINING.LAMBDA_REC + self.cycle_loss * _MANIFEST.TRAINING.LAMBDA_CYCLE / 2 + self.ee_loss * _MANIFEST.TRAINING.LAMBDA_EE / 2 + self.generator_loss*1

        self.total_generator_loss.backward()

    def _backward_discriminator_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
               Parameters:
                   netD (network)      -- the discriminator D
                   real (tensor array) -- real images
                   fake (tensor array) -- images generated by a generator
               Return the discriminator loss.
               We also call loss_D.backward() to calculate the gradients.
               """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def _backward_discriminator(self):
        self.discriminator_losses = []
        self.discriminator_loss = 0

        for i in range(self._number_of_groups):
            fake = self.fake_pools[i].query(self.fake_positions[2-i])
            self.discriminator_losses.append(self._backward_discriminator_basic(self.nn_models[i].discriminator,
                                                                                self.positions_reference[i].detach(),
                                                                                fake))
            self.discriminator_loss += self.discriminator_losses[-1]
            self.loss_recorder.add_scalar('discriminator_loss_{}'.format(i), self.discriminator_losses[-1].clone().detach())
    def load(self, epoch=None):

        for i, model in enumerate(self.nn_models):
            model.load(epoch)

        if _MANIFEST.TRAINING.IS_TRAINING:
            for i, optimizer in enumerate(self.optimizers):
                filepath = os.path.join(_MANIFEST.TRAINING.MODELS_DIR, 'optimizers/{}/{}.pt'.format(epoch,i))
                optimizer.load_state_dict(load(filepath, weights_only=True))


    def save(self, epoch=None):
        '''
        '''
        # save the models, one for each skeleton type in training
        for model in self.nn_models:
            model.save(epoch)

        # save the optimizers, 0.pt = generator optimizer, 1.pt = discriminator optimizer
        optimizers_path = '{}optimizers/{}/'.format(_MANIFEST.TRAINING.MODELS_DIR, epoch)
        if not os.path.exists(optimizers_path):
            os.makedirs(optimizers_path, exist_ok=True)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(optimizers_path, '{}.pt'.format(i))
            save(optimizer.state_dict(), file_name)


    def set_input(self, motions):
        '''

        :param motions: dict
        :param is_training: bool - from manifest IS_TRAINING flag
        :return:
        '''
        is_training = _MANIFEST.TRAINING.IS_TRAINING

        self._motions = motions

        if not is_training:
            self._motions_backup = []
            for i in range(2):
                self._motions_backup.append(motions[i][0].clone())
                self._motions[i][0][1:] = self._motions[i][0][0]
                self._motions[i][1] = [0] * len(self._motions[i][1])

    def optimize_parameters(self):
        gan_mode = _MANIFEST.MODEL.GAN_MODE

        self._forward()

        # update generators
        self._set_discriminator_grad_requirement(False)
        self._generator_optimizer.zero_grad()
        self._backward_generator()
        self._generator_optimizer.step()

        # update discriminators
        if gan_mode == 'lsgan':
            self._set_discriminator_grad_requirement(True)
            self._discriminator_optimizer.zero_grad()
            self._backward_discriminator()
            self._discriminator_optimizer.step()
        elif gan_mode == 'default':
            self.discriminator_loss = tensor(0)
        else:
            raise NotImplementedError('GANModel does not support gan_mode={}'.format(gan_mode))

    def verbose(self):
        '''
        '''
        result = {
            'recorded_loss_0': self.recorded_losses[0].item(),
            'recorded_loss_1': self.recorded_losses[1].item(),
            'cycle_loss': self.cycle_loss.item(),
            'ee_loss': self.ee_loss.item(),
            'discriminator_loss_gan': self.discriminator_loss.item(),
            'generator_loss_gan': self.generator_loss.item()
        }

        return sorted(result.items(),key=lambda x:x[0])

    def _compute_test_result(self):
        gt_poses = []
        gt_denorm = []

        for src in range(self._number_of_groups):
            gt = self._motions_backup[src]
            index = list(range(gt.shape[0]))
            gt = self.dataset.denormalize(src,index,gt)
            gt_denorm.append(gt)
            gt_pose = self.nn_models[src].fk.forward_from_raw(gt, self.dataset.offsets[src][index])
            gt_poses.append(gt_pose)

            for i in index:
                new_path = os.path.join(self.bvh_path, self.characters_by_group[src][i]['character'])
                if not os.path.exists(new_path):
                    os.makedirs(new_path, exist_ok=True)
                self.writer[src][i].write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))

        p = 0
        for src in range(self._number_of_groups):
            for dst in range(self._number_of_groups):
                for i in range(len(self.characters_by_group[dst])):
                    dst_path = os.path.join(self.bvh_path, self.characters_by_group[dst][i]['character'])
                    self.writer[dst][i].write_raw(self.fake_results_denormalized[p][i, ...], 'quaternion',
                                                  os.path.join(dst_path, '{}_{}.bvh'.format(self.id_test, src)))
                p += 1
        self.id_test += 1


