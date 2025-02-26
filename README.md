Official PyTorch implementation of the paper "Efficient Masked AutoEncoder for Video Object Counting and A Large-Scale Benchmark".

# DroneBird dataset
## Download
百度网盘：
[Link](https://pan.baidu.com/s/1UZ34XcqKMiVMZUuTVV3EpQ?pwd=t3tv)
提取码：t3tv

## Evaluation
put the result `.txt` file on the root of dataset and run the script `eval.py` in `toolkit` folder.

# E-MAC
## Setup
### Create conda environment
create conda environment from sepc_list.txt
```bash
conda create --name <env_name> --file spec_list.txt
```

### Install pwcnet
Follow the [instructions](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/README.md) to install pwcnet in folder './emac/models'

The folder structure should be like:
```
emac
└── models
    ├── correlation_package
    ├── __init__.py
    └── pwcnet.py
```

### Download the pretrained model
Download the pretrained weight of `MultiMAE` and `PWC-Net` to `./cfgs`

## Training
```bash
bash density_opt.sh
```
Path of the config file should be set in `density_opt.sh`.

## Inference
```bash
python test_opt.py
```

Some args in `test_opt.py`:
- 'image_height, image_width': the size of input images
- 'data_path': the path to the dataset
- 'dataset': the dataset name
- 'weight_path': the path to the model weights

# Citation
```bibtex
@inproceedings{
cao2025efficient,
title={Efficient Masked AutoEncoder for Video Object Counting and A Large-Scale Benchmark},
author={Bing Cao and Quanhao Lu and Jiekang Feng and Qilong Wang and Pengfei Zhu and Qinghua Hu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=sY3anJ8C68}
}
```

# Acknowledgement
This code is based on the [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) and [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master). We thank the authors for their excellent work.