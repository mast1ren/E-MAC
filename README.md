Official implementation of paper Efficient Masked Autoencoder for Video Object Counting and A Large-Scale Benchmark.

# DroneBird dataset
## Download
TBD
## Evaluation

# E-MAC
## Setup
### create conda environment
create conda environment from sepc_list.txt
```bash
conda create --name <env_name> --file spec_list.txt
```

### install pwcnet
Follow the [instructions](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/README.md) to install pwcnet in folder './emac/models'

## Inference
```bash
python test_opt.py
```

Some args in `test_opt.py`:
- 'image_height, image_width': the size of input images
- 'data_path': the path to the dataset
- 'dataset': the dataset name
- 'weight_path': the path to the model weights
