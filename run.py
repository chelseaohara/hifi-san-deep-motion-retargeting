from config.manifest import load_manifest, set_manifest
from model.gan_model import GANModel
from utils.data_manager import DataManager
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time


if __name__ == '__main__':
    # get default values from manifest.py YACS config
    manifest = load_manifest()
    # merge with selected experiment
    manifest.merge_from_file('./config/exp_template.yaml')
    manifest.freeze()
    # set merged manifest object as manifest to reference throughout program
    set_manifest(manifest)

    if manifest.SYSTEM.USE_IPDB:
        import ipdb
        ipdb.set_trace()

    print("\n\n","Executing Experiment: ", manifest.TRAINING.EXPERIMENT_NAME, "\n","-"*54,"\n\n")

    # initialize Data Manager
    DM = DataManager()

    # preprocess data for dataset creation
    DM.preprocess_data()

    # create dataset for experiment
    skeleton_dataset = DM.create_dataset(manifest.DATA.GROUP_A, manifest.DATA.GROUP_B)

    print("Begin training.")

    # load data from dataset with Torch DataLoader
    data_loader = DataLoader(skeleton_dataset,
                             batch_size=manifest.TRAINING.BATCH_SIZE,
                             shuffle=manifest.TRAINING.SHUFFLE,
                             num_workers=manifest.TRAINING.NUMBER_OF_WORKERS,
                             prefetch_factor=manifest.TRAINING.PREFETCH_FACTOR,
                             pin_memory=manifest.TRAINING.PIN_MEMORY,
                             persistent_workers=manifest.TRAINING.PERSISTENT_WORKERS
                             )

    gan_model = GANModel(skeleton_dataset)

    gan_model.setup()

    start_time = time.time()

    for epoch in range(0, manifest.TRAINING.NUMBER_OF_EPOCHS+1):
        for step, motions in tqdm(enumerate(data_loader),total=len(data_loader)):
            gan_model.set_input(motions)
            gan_model.optimize_parameters()

        if epoch % manifest.TRAINING.INTERVAL == 0 or epoch == manifest.TRAINING.NUMBER_OF_EPOCHS:
            gan_model.save(epoch=epoch)

        gan_model.epoch()

    end_time = time.time()
    print("Total Training Time: {}".format(end_time - start_time))