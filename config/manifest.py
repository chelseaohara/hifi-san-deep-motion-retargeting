'''
Manifest
---
Holds information relevant to current training/execution model. Values can be overridden by YAML files.
'''

from yacs.config import CfgNode as ConfigurationNode
from torch import cuda

_C = ConfigurationNode()

# Helper Functions
def _get_device():
    if cuda.is_available():
        print('CUDA IS AVAILABLE. USING GPU TO TRAIN/TEST.')
        return 'cuda'
    else:
        print('CUDA N/A. USING CPU TO TRAIN.')
        return 'cpu'

# ---- PROJECT CONFIG ---------------
_C.SYSTEM = ConfigurationNode()
_C.SYSTEM.DEVICE = _get_device()
_C.SYSTEM.USE_IPDB = False

_C.TRAINING = ConfigurationNode()
_C.TRAINING.EXPERIMENT_NAME = 'default_training_experiment'
_C.TRAINING.IS_TRAINING = True

_C.TRAINING.EPOCH_BEGIN = 0
_C.TRAINING.NUMBER_OF_EPOCHS = 100
_C.TRAINING.INTERVAL = 50
_C.TRAINING.BATCH_SIZE = 128
_C.TRAINING.SHUFFLE = True
_C.TRAINING.NUMBER_OF_WORKERS = 2
_C.TRAINING.PREFETCH_FACTOR = 2
_C.TRAINING.PIN_MEMORY = True
_C.TRAINING.PERSISTENT_WORKERS = True
_C.TRAINING.OPERATOR = 'simple'
_C.TRAINING.KERNEL_SIZE = 15 # "must be odd"
_C.TRAINING.DEGREE_OF_SEPARATION = 2
_C.TRAINING.MODELS_DIR = './results/models/'
_C.TRAINING.LAMBDA_GLOBAL_POSE = 2.5
_C.TRAINING.LAMBDA_POSITION = 1
_C.TRAINING.LAMBDA_REC = 5
_C.TRAINING.LAMBDA_CYCLE = 5
_C.TRAINING.LAMBDA_EE = 100

_C.MODEL = ConfigurationNode()
_C.MODEL.LOGS_DIR = './results/logs'
_C.MODEL.NUMBER_OF_LAYERS = 2
_C.MODEL.GAN_MODE = 'default'
# learning rate params
_C.MODEL.LEARNING_RATE = 3e-4
_C.MODEL.SCHEDULER = 'default'
_C.MODEL.LR_STEP_SIZE = 50
_C.MODEL.LR_STEP_GAMMA = 0.5
_C.MODEL.LR_PLAT_MODE = 'min'
_C.MODEL.LR_PLAT_FACTOR = 0.2
_C.MODEL.LR_PLAT_THRESHOLD = 0.01
_C.MODEL.LR_PLAT_PATIENCE = 5
_C.MODEL.LR_PLAT_VERBOSE = True
_C.MODEL.LR_MULTISTEP_MILESTONES = []
# NOTE: these two values are included in code but not defined in option_parser.py
_C.MODEL.N_EPOCHS_ORIGIN = 0
_C.MODEL.N_EPOCHS_DECAY = 0
# IntegratedModel Config
_C.MODEL.IM_USE_SEPARATE_END_EFFECTORS = False
_C.MODEL.IM_END_EFFECTOR_LOSS_FACT = 'learn'
_C.MODEL.POOL_SIZE = 50
# AutoEncoder / Encoder / Decoder
_C.MODEL.LATSZ = 128 # size of latent space (for rotation encoding)
# Discriminator
_C.MODEL.IS_PATCH_GAN = True

# Configurations for the dataset
_C.DATA = ConfigurationNode()
# Flag to preprocess raw data for training
_C.DATA.PREPROCESS = True
# raw data directory
_C.DATA.RAW_DIR = './data/raw'
_C.DATA.RAW_DIR_NOISY = './data/raw/_noisy'
# prepared data directory
_C.DATA.PREPARED_DIR = './data/prepared'
_C.DATA.MEAN_VAR_DIR = './data/mean_var'
_C.DATA.REF_BVH_DIR = './data/reference'
# number of groups (may only be 2, A and B)
_C.DATA.NUMBER_OF_GROUPS = 2
# group names list
_C.DATA.GROUPS = ['GROUP_A', 'GROUP_B']
# List of characters in Group A by name
_C.DATA.GROUP_A = []
# List of characters in Group B by name
_C.DATA.GROUP_B = []
_C.DATA.AUGMENT = False
_C.DATA.NORMALIZE = True
_C.DATA.SKELETON_ATTRIBUTES = []
# Length of time axis (number of frames * delta time) per window
_C.DATA.WINDOW_SIZE = 28
_C.DATA.ORDER = 'xyz'
_C.DATA.ROTATION_TYPE = 'quaternion'
_C.DATA.POSITION_REPRESENTATION = '3D'
_C.DATA.SKELETON_INFO_HANDLING = 'concat'
_C.DATA.WORLD = False
# End Effectors
_C.DATA.EE_VELO = True
_C.DATA.EE_FROM_ROOT = True
# Params for noising
_C.DATA.NOISE = 2.00
_C.DATA.DROPOUT_PROBABILITY =0.001


# TESTING MODEL PARAMS
_C.TEST = ConfigurationNode()
TEST_NAME = ''
_C.TEST.OUTPUT_DIR = './results/retargeting'
_C.TEST.CHARACTERS = []
_C.TEST.NUMBER_OF_EPOCHS = 0
_C.TEST.GROUPS = ['GROUP_A', 'GROUP_B']
# List of characters in Group A by name
_C.TEST.GROUP_A = []
# List of characters in Group B by name
_C.TEST.GROUP_B = []
_C.TEST.CHARTYPE = ''
# For testing noisy data
_C.TEST.USE_DIRTY = False


# ---- MANIFEST TOOLS ----------------
# Manifest of the experiment; accessed through get_manifest()
manifest = _C

# Accessor Functions
def get_manifest():
    '''
    Called throughout program to provide immutable values of manifest
    :return: manifest object
    '''
    return manifest

def set_manifest(new_manifest):
    '''
    Setter function to assign the manifest for use
    :return:
    '''
    manifest.update(new_manifest)

def load_manifest():
    '''
    Called on program entry to provide defaults values of the manifest for training
    :return: clone of CfgNode object (so that defaults will not be altered).
    '''
    return _C.clone()