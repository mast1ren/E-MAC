Official PyTorch implementation of the paper "Efficient Masked AutoEncoder for Video Object Counting and A Large-Scale Benchmark".

# DroneBird dataset
## Download
百度网盘：
[Link](https://pan.baidu.com/s/1UZ34XcqKMiVMZUuTVV3EpQ?pwd=t3tv)
提取码：t3tv
Google Drive:
Coming soon

## Evaluation
put the result `.txt` file on the root of dataset and run the script `eval.py` in `toolkit` folder.

# E-MAC
## Setup
### Create conda environment

create conda environment by
```bash
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```


### Install pwcnet
Follow the [instructions](https://github.com/NVlabs/PWC-Net) to install pwcnet in folder './emac/models'

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

# License
 <p xmlns:cc="http://creativecommons.org/ns#" >This work (dataset and code) is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p> 
 