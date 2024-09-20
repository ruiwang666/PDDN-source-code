# PDDN-source-code

**Folder: "Stage1", "Stage2" ,and "VAE" are the modules of PDDN**

**For testing purpose, please run souce code with radar data by following steps:**

* Process 'train_2d_autoencoder.py' and 'train_3d_autoencoder.py' in the 'var' folder to produce the pretrained model for the 2d autoencoder (used in stage1 diffusion) and 3d autoencoder (used in the stage2 diffusion).

* Process 'PDDN.py' in the 'Stage1' folder to produce the pretrained model for the 3d U-Net (to be used in the Stage2)

* Process 'PDDN.py' in the 'Stage2' folder to produce the pretrained model for the 3d U-Net of the Stage2

## Packages Requirement

**The PDDN is developed based on Pytorch, Torch Lighting, and MONAI libraries.**

**The autoencoder used in PDDN is built based on the code of 'prediff' (https://github.com/gaozhihan/PreDiff) and 'ldcast' (https://github.com/MeteoSwiss/ldcast).**

**Run 'conda env create -f environment.yml' to install the packages.**

**All code was developed and tested on 8 NVIDIA H800 GPU.**

**The training data is HKO Dopper Radar reflectivity data, provided by the Hong Kong Observatory. Please contact HKO directly for the permission to use the radar data.**

**Please contact Email: rwangbp@connect.ust.hk if you have any concerns using this code.**

## Copyright Statement
The code in this project is written by Rui Wang, currently PhD student at HKUST supervised by professor Jimmy Fung. This code is only for testing purpose by reviewers of our submitted manuscript. Any actions of copying the code for publication or commercial usage before the formal publication of our code related manuscript will be considered as copyright violation.

## Acknowledgment
We appreciate the assistance of the Hong Kong Observatory (HKO), which provided the meteorological data. The work described in this paper was supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project Nos. AoE/E-603/18, and T31-603/21-N).
