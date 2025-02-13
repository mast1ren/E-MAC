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
