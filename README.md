# High-Fidelity Skeleton-Aware Networks for Deep Motion Retargeting

### Status: Under Review

## Overview

Codebase for the Winter-2025 USRA + Fall-2025 USRA project research by Chelsea O'Hara (Faculty of Engineering and Applied 
Science, Memorial University of Newfoundland), supervised by Dr. Robert Gallant (School of Science and the Environment, 
Grenfell Campus, Memorial University of Newfoundland). The companion paper <em>High Fidelity Skeleton Aware Networks</em> 
can be viewed [here](https://drive.google.com/file/d/1y30lgMuDMx4ZqmiDKHibZn9Q3e-33RA5/view?usp=sharing). A supplementary video is also available on [YouTube](https://www.youtube.com/watch?v=eqghurbjf1c).

Building upon the Skeleton-Aware Network (SAN) framework established by [Aberman et al.](https://github.com/DeepMotionEditing/deep-motion-editing), 
which utilizes skeletal pooling to reduce homeomorphic graphs into a common latent space, we propose 
**High-Fidelity Skeleton-Aware Networks (HiFi SAN)**. Our architecture reimagines the cycle-consistent framework by 
replacing skeletal convolution layers with Transformer blocks to leverage Self-Attention and effectively capture 
global spatial-temporal dependencies across the kinematic chain.

![hifi-san-preview.gif](media/hifi-san-preview.gif)

## Getting Started

### Packages and Versions

Package versions were chosen based on compatability across multiple environments. These are not necessarily required, 
or the only compatible packages, but are listed here to document the environment configuration used to produce the 
results in the paper.

- Python [3.10]
- PyTorch [2.5.1]
- TorchVision [0.20.1]
- TorchAudio [2.5.1]
- PyTorch Cuda [12.4]
- Tensorboard [2.19.0]
- tqdm [4.67.1]
- NumPy [2.1.3]
- YACS [0.1.8]

### Data Collection

Data for this project was curated from the [Mixamo](https://www.mixamo.com/) library as FBX files saved at 60 fps without skin. The `data\raw\_fbx` 
directory was used to store FBX files for each character to be converted to BVH files using the script in `data\fbx_converter.py`
(Note that the fbx_converter.py must be placed in the `.\blender` (Blender 3) install directory to work).

BVH files may be procured from alternative sources, though consideration would need to be taken for the skeleton definitions 
in the Manifest YAMLs. Configuration file templates are provided.

The data directory tree should be similar to the following:
```
.
├── mean_var
│   ├── Character01.npy
│   ├── Character01.npy
│   └── ...
├── prepared
│   ├── Character01.npy
│   ├── Character02.npy
│   └── ...
├── raw
│   ├── _fbx
│   │   ├── Character01
│   │   │   ├── Character01_Acknowledging.fbx   <--- Note the name structure for the fbx_converter.py tool
│   │   │   └── ...
│   │   ├── Character02
│   │   │   ├── Character02_Acknowledging.fbx 
│   │   │   └── ...
│   │   └── ...
│   ├── _noisy
│   ├── Character01                             <--- BVH files are stored per character at this level
│   │   │   ├── Character01_Acknowledging.bvh 
│   │   │   └── ...
│   ├── Character02
│   │   │   ├── Character02_Acknowledging.bvh 
│   │   │   └── ...
│   │   └── ...
├── reference
│   ├── Character01_std.bvh
│   ├── Character02_std.bvh
│   └── ...
├── fbx_converter.py
├── make_noisy.py
├── mixed_data.py
├── motion_dataset.py
├── skeleton_dataset.py
└── test_dataset.py
```

## Training
First, verify that all necessary packages are installed. Note that IPython pdb is optional and can be enabled in the 
`config/manifest.py` file by setting `_C.SYSTEM.USE_IPDB` to `True`. Then, configure the following files and directories
 as so:

1. Copy or edit the sample experiment yaml `config/exp_template.yaml` file to include the character names used in the dataset
2. Update `skeletons.yaml` with the list of bones and end effectors for each skeleton domain (a sample is included in `skeletons_reference.yaml`)
3. Place a directory for each character in `data/raw`. Directory names are case-sensitive to character and should contain BVH files only
4. Make sure that `data/mean_var`, `data/prepared`, and `data/reference` directories exist
5. Edit the Manifest with `config/manifest.yaml` or override any values in the Manifest for your training run by adding them to the experiment yaml
6. Update `run.py` to use appropriate experiment yaml (ex. `manifest.merge_from_file('./config/exp_template.yaml'`)

### Running the project
From the terminal in the project directory
```commandline
python run.py
```

## Testing
Testing is done using the `results/results_stats.py` file. Tests can be done using the training dataset motions or on 
motions the model has not seen before.

1. Copy or edit the `config/test_template.yaml` or `config/test_unseen.yaml` to include the appropriate test parameters. These are where values from the Manifest can be overwritten
2. Set the test_name parameter -- this will be used as the output directory for BVH files written from test results
3. Add the BVH motion file names to `motions.yaml` or `motions_unseen.yaml`
3. Update `results/results_stats.py` to include the appropriate test yaml (ex. `manifest.merge_from_file('./config/test_template.yaml'`)

### Running the test
From the terminal in the project directory
```commandline
python results/results_stats.py
```

## Authors & Acknowledgments
* **Chelsea O'Hara** - [GitHub](https://github.com/chelseaohara) // Email [chelsea.ohara@mun.ca](mailto:chelsea.ohara@mun.ca) (Primary Contact)
* **Robert Gallant** - Email [h79rpg@mun.ca](mailto:h79rpg@mun.ca)
* This research was supported by the **Natural Sciences and Engineering Research Council of Canada Undergraduate Student Research Awards (USRA)** program and administered by Memorial University of Newfoundland.

## License & Attribution

This project is licensed under the BSD 2-Clause License.

### Credits
* This codebase is a refactor and modernization of the framework originally proposed by **Aberman et al. (2020)** in *"Skeleton-Aware Networks for Deep Motion Retargeting"*.
* The BVH parsing logic is adapted from the work of **Daniel Holden**.
* Replay buffer logic is adapted from **Zhu et al. (CycleGAN)**.

See the [LICENSE](LICENSE.md) file for the full text and original copyright notices.