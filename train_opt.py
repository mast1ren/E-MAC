# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Union, Tuple
from utils.log_images import log_image_wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from einops import rearrange, repeat
import utils
import utils.data_constants as data_constants
from emac.emac_utils import warp, TransFuse
from emac import emac
from emac.criterion import (
    MaskedMSELoss,
    TVLoss,
)
from emac.input_adapters import (
    PatchedInputAdapter,
)
from emac.output_adapters import (
    SpatialOutputAdapter,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import DICT_MEAN_STD
from utils.density.density_dataset import (
    buildDensityDataset,
)
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.task_balancing import NoWeightingStrategy, UncertaintyWeightingStrategy
from utils.pos_embed import interpolate_pos_embed_multimae


DOMAIN_CONF = {
    "rgb": {
        "channels": 3,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=3),
        "loss": MaskedMSELoss,
    },
    "density": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
        "loss": MaskedMSELoss,
    },
}


@torch.no_grad()
def make_density_metrics(preds, target):
    b, c, _, _ = preds.shape
    n = b
    diff = torch.abs(preds.sum(dim=[1, 2, 3]) - target.sum(dim=[1, 2, 3]))

    metrics = {
        "err": diff.sum() / n,
    }
    return metrics


def get_args():
    config_parser = parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser("MultiMAE pre-training script", add_help=False)

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size per GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        default=1600,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        default=10,
        type=int,
        help="Checkpoint saving frequency in epochs (default: %(default)s)",
    )
    parser.add_argument("--save_ckpt", action="store_true")
    parser.set_defaults(save_ckpt=True)
    parser.add_argument(
        "--eval_freq",
        default=1,
        type=int,
        help="Evaluation frequency in epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_first",
        default=False,
        action="store_true",
        help="Evaluate model before training",
    )
    parser.add_argument(
        "--max_train_images", default=1000, type=int, help="number of train images"
    )
    parser.add_argument(
        "--max_val_images", default=100, type=int, help="number of validation images"
    )
    parser.add_argument(
        "--max_test_images", default=54514, type=int, help="number of test images"
    )

    parser.add_argument(
        "--in_domains",
        default="rgb-density-rgb",
        type=str,
        help="Input domain names, separated by hyphen (default: %(default)s)",
    )
    parser.add_argument(
        "--out_domains",
        default="rgb-density-rgb",
        type=str,
        help="Output domain names, separated by hyphen (default: %(default)s)",
    )
    parser.add_argument("--standardize_depth", action="store_true")
    parser.add_argument(
        "--no_standardize_depth", action="store_false", dest="standardize_depth"
    )
    parser.set_defaults(standardize_depth=False)
    parser.add_argument("--extra_norm_pix_loss", action="store_true")
    parser.add_argument("--extra_unnorm_den_loss", action="store_true")
    parser.add_argument(
        "--no_extra_norm_pix_loss", action="store_false", dest="extra_norm_pix_loss"
    )
    parser.set_defaults(extra_norm_pix_loss=True)
    parser.add_argument("--use_opt_loss", default=True, action="store_true")
    parser.add_argument("--use_cur_loss", default=True, action="store_true")
    parser.add_argument("--use_tv_loss", default=True, action="store_true")
    parser.add_argument(
        "--loss_weights", default={"opt": 1.0, "cur": 10.0, "tv": 20.0, "fus": 10.0}
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="pretrain_multimae_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train (default: %(default)s)",
    )
    parser.add_argument(
        "--is_mask_inputs",
        default=False,
        type=bool,
        help="Set to True/False to enable/disable masking of input tokens",
    )
    parser.add_argument(
        "--total_num_tokens",
        default=400,
        type=int,
        help="Total number of tokens in the model (default: %(default)s)",
    )
    parser.add_argument(
        "--num_encoded_tokens",
        default=98,
        type=int,
        help="Number of tokens to randomly choose for encoder (default: %(default)s)",
    )
    parser.add_argument(
        "--num_global_tokens",
        default=1,
        type=int,
        help="Number of global tokens to add to encoder (default: %(default)s)",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Base patch size for image-like modalities (default: %(default)s)",
    )
    parser.add_argument(
        "--input_size",
        default=224,
        type=Union[int, Tuple[int]],
        help="Images input size for backbone (default: %(default)s)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        default=1.0,
        help="Dirichlet alphas concentration parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--sample_tasks_uniformly",
        default=False,
        action="store_true",
        help="Set to True/False to enable/disable uniform sampling over tasks to sample masks for.",
    )

    parser.add_argument(
        "--decoder_use_task_queries",
        default=True,
        action="store_true",
        help="Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens",
    )
    parser.add_argument(
        "--decoder_use_xattn",
        default=True,
        action="store_true",
        help="Set to True/False to enable/disable decoder cross attention.",
    )
    parser.add_argument(
        "--decoder_dim",
        default=256,
        type=int,
        help="Token dimension inside the decoder layers (default: %(default)s)",
    )
    parser.add_argument(
        "--decoder_depth",
        default=2,
        type=int,
        help="Number of self-attention layers after the initial cross attention (default: %(default)s)",
    )
    parser.add_argument(
        "--decoder_num_heads",
        default=8,
        type=int,
        help="Number of attention heads in decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: %(default)s)",
    )

    parser.add_argument(
        "--loss_on_unmasked",
        default=False,
        action="store_true",
        help="Set to True/False to enable/disable computing the loss on non-masked tokens",
    )
    parser.add_argument(
        "--no_loss_on_unmasked", action="store_false", dest="loss_on_unmasked"
    )
    parser.set_defaults(loss_on_unmasked=True)

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help="Optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--opt_betas",
        default=[0.9, 0.95],
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer betas (default: %(default)s)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="CLIPNORM",
        help="Clip gradient norm (default: %(default)s)",
    )
    parser.add_argument(
        "--skip_grad",
        type=float,
        default=None,
        metavar="SKIPNORM",
        help="Skip update if gradient norm larger than threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""",
    )
    parser.add_argument(
        "--decoder_decay", type=float, default=None, help="decoder weight decay"
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="Warmup learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--task_balancer",
        type=str,
        default="none",
        help="Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)",
    )
    parser.add_argument(
        "--balancer_lr_scale",
        type=float,
        default=1.0,
        help="Task loss balancer LR scale (if used) (default: %(default)s)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=40,
        metavar="N",
        help="Epochs to warmup LR, if scheduler supports (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="Epochs to warmup LR, if scheduler supports (default: %(default)s)",
    )

    parser.add_argument(
        "--fp32_output_adapters",
        type=str,
        default="",
        help="Tasks output adapters to compute in fp32 mode, separated by hyphen.",
    )

    # Augmentation parameters
    parser.add_argument(
        "--hflip",
        type=float,
        default=0.5,
        help="Probability of horizontal flip (default: %(default)s)",
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help="Training interpolation (random, bilinear, bicubic) (default: %(default)s)",
    )
    parser.add_argument(
        "--cilp_size",
        type=int,
        default=5,
        help="Probability of horizontal flip (default: %(default)s)",
    )
    # Dataset parameters
    parser.add_argument(
        "--dataset",
        default="FDST",
        type=str,
        # metavar='DATASET',
        help="Name of dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--data_path",
        default=data_constants.IMAGENET_TRAIN_PATH,
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )
    parser.add_argument(
        "--data_clip_size",
        default=2,
        type=int,
        help="Number of frames to clip from the video",
    )
    parser.add_argument(
        "--data_stride",
        default=1,
        type=int,
        help="Stride for the clip",
    )
    parser.add_argument(
        "--DATA_MEAN",
        default=None,
        type=float,
        help="Mean of the dataset",
    )
    parser.add_argument(
        "--DATA_STD",
        default=None,
        type=float,
        help="Standard deviation of the dataset",
    )

    # Misc.
    parser.add_argument(
        "--output_dir", default="", help="Path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed ")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument(
        "--no_find_unused_params", action="store_false", dest="find_unused_params"
    )
    parser.set_defaults(find_unused_params=True)
    parser.add_argument(
        "--train_print_freq",
        default=10,
        type=int,
        help="Logging frequency in steps (default: %(default)s)",
    )
    parser.add_argument(
        "--val_print_freq",
        default=50,
        type=int,
        help="Logging frequency in steps (default: %(default)s) in evaluation",
    )

    # Wandb logging
    parser.add_argument(
        "--log_wandb",
        default=False,
        action="store_true",
        help="Log training and validation metrics to wandb",
    )
    parser.add_argument("--no_log_wandb", action="store_false", dest="log_wandb")
    parser.set_defaults(log_wandb=False)
    parser.add_argument(
        "--wandb_project", default=None, type=str, help="Project name on wandb"
    )
    parser.add_argument(
        "--wandb_entity", default=None, type=str, help="User or team name on wandb"
    )
    parser.add_argument(
        "--wandb_run_name", default=None, type=str, help="Run name on wandb"
    )
    parser.add_argument("--log_images_wandb", action="store_true")
    parser.add_argument(
        "--log_images_freq",
        default=5,
        type=int,
        help="Frequency of image logging (in epochs)",
    )
    parser.add_argument("--show_user_warnings", default=False, action="store_true")

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--ckpt_multi", default=None, type=str, help="path to checkpoint to load"
    )

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    if args.DATA_MEAN is None or args.DATA_STD is None:
        args.DATA_MEAN, args.DATA_STD = (
            DICT_MEAN_STD[args.dataset]["MEAN"],
            DICT_MEAN_STD[args.dataset]["STD"],
        )

    return args


def get_model(args):
    """Creates and returns model from arguments"""
    print(
        f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}"
    )

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
            # image_size=(448, 640),
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]["output_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
            # image_size=(448, 640),
            num_channels=1,
            dim_tokens_enc=768,
            # reg_counts=True,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
        )
        for domain in args.out_domains
    }

    fuse_module = TransFuse(
        stride_level=1,
        patch_size_full=args.patch_size,
        image_size=args.input_size,
        num_channels=1,
        dim_tokens_enc=768,
        dim_tokens=args.decoder_dim,
        num_heads=args.decoder_num_heads,
    )

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters["norm_rgb"] = DOMAIN_CONF["rgb"]["output_adapter"](
            stride_level=DOMAIN_CONF["rgb"]["stride_level"],
            patch_size_full=args.patch_size,
        )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        fuse_module=fuse_module,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
    )
    print("model created")

    return model


def main(args):
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    print("ori seed = %d" % args.seed)
    seed = args.seed + utils.get_rank()
    print("seed = %d" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    print(torch.initial_seed())
    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split("-")
    args.out_domains = args.out_domains.split("-")
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)
    # for k, v in model.named_parameters():
    #     print(k, v.shape)

    if args.task_balancer == "uncertainty":
        loss_balancer = UncertaintyWeightingStrategy(tasks=["density", "count"])
    else:
        loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]["loss"](
            patch_size=args.patch_size,
            stride=DOMAIN_CONF[domain]["stride_level"],
        )
        for domain in args.out_domains
    }
    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](
            patch_size=args.patch_size,
            stride=DOMAIN_CONF["rgb"]["stride_level"],
            norm_pix=True,
        )
    print("loss loaded")

    # Get dataset
    dataset_train = buildDensityDataset(
        data_root=args.data_path,
        tasks=args.all_domains,
        split="train",
        variant="tiny",
        image_size=args.input_size,
        max_images=args.max_train_images,
        dataset_name=args.dataset,
        clip_size=args.data_clip_size,
        stride=args.data_stride,
        MEAN=[args.DATA_MEAN],
        STD=[args.DATA_STD],
    )

    dataset_val = buildDensityDataset(
        data_root=args.data_path,
        tasks=args.all_domains,
        split="val",
        variant="tiny",
        image_size=args.input_size,
        max_images=args.max_val_images,
        dataset_name=args.dataset,
        clip_size=args.data_clip_size,
        stride=args.data_stride,
        MEAN=[args.DATA_MEAN],
        STD=[args.DATA_STD],
    )

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = (
            len(dataset_train) // args.batch_size // num_tasks
        )
        print(len(dataset_train), args.batch_size, num_tasks)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=sampler_rank,
            shuffle=True,
            drop_last=True,
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # collate_fn=train_collate,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    if args.ckpt_multi is not None:
        checkpoint = torch.load(args.ckpt_multi, map_location="cpu")

        checkpoint_model = checkpoint["model"]

        # Load pre-trained model
        # # Remove keys for semantic segmentation
        for k in list(checkpoint_model.keys()):
            if "semseg" in k:
                del checkpoint_model[k]
        # Remove output adapters
        for k in list(checkpoint_model.keys()):
            if (
                "output_adapters" in k
                and "output_adapters.rgb" not in k
                and "output_adapters.density" not in k
            ):
                del checkpoint_model[k]
        copy_layers = []

        # Interpolate position embedding
        interpolate_pos_embed_multimae(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        # msg = model.video_encoder.load_state_dict(video_ckpt['model'], strict=False)
        # print(msg)

        model.state_dict()["input_adapters.density.proj.weight"].copy_(
            checkpoint_model["input_adapters.rgb.proj.weight"].sum(1, keepdim=True)
        )
        model.state_dict()["input_adapters.density.pos_emb"].copy_(
            checkpoint_model["input_adapters.rgb.pos_emb"]
        )
        # print(model.state_dict()['input_adapters.density.pos_emb'].shape)
        model.state_dict()["output_adapters.density.pos_emb"].copy_(
            rearrange(
                checkpoint_model["input_adapters.rgb.pos_emb"],
                "b (d c) h w  -> b d c h w",
                c=3,
            ).sum(2)
            # checkpoint_model['input_adapters.rgb.pos_emb']
        )

        model.state_dict()["output_adapters.density.out_proj.weight"].copy_(
            checkpoint_model["output_adapters.rgb.out_proj.weight"]
            .reshape(256, 3, -1)
            .sum(1, keepdim=True)
            .reshape(256, -1)
            # .repeat(1, 2)
        )

        copy_layers.append("input_adapters.density.pos_emb")

        i = 0
        for k in list(model.state_dict().keys()):
            if (
                "output_adapters.density" in k
                and "out_proj" not in k
                and "pos_emb" not in k
                and "addconv" not in k
                and "smoothconv" not in k
            ):
                i += 1
                print(k)
                model.state_dict()[k].copy_(
                    checkpoint_model[k.replace("density", "rgb")]
                )
                copy_layers.append(k)

        model.fuse.state_dict()["out_proj.weight"].copy_(
            checkpoint_model["output_adapters.rgb.out_proj.weight"]
            .reshape(256, 3, -1)
            .sum(1, keepdim=True)
            .reshape(256, -1)
        )
        model.fuse.state_dict()["pos_emb"].copy_(
            rearrange(
                checkpoint_model["input_adapters.rgb.pos_emb"],
                "b (d c) h w  -> b d c h w",
                c=3,
            ).sum(2)
        )
        for k in list(model.fuse.state_dict().keys()):
            if "out_proj" not in k and "pos_emb" not in k:
                model.fuse.state_dict()[k].copy_(
                    checkpoint_model["output_adapters.rgb." + k]
                )
        print(copy_layers)
        print(i)

        model.pwc.load_state_dict(torch.load("./cfgs/pwc_net.pth.tar"))
    model.to(device)
    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        # idx=0: input adapters, idx>0: transformer layers
        layer_decay_values = list(
            args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
        )
        assigner = LayerDecayValueAssigner(layer_decay_values)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module

    if args.distributed and args.task_balancer != "none":
        loss_balancer = torch.nn.parallel.DistributedDataParallel(
            loss_balancer, device_ids=[args.gpu]
        )
        loss_balancer_without_ddp = loss_balancer.module

    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None,
    )
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    # print(wd_schedule_values)
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_val_loss = np.inf
    best_epoch = 0
    if args.eval_first:
        val_stats, mae, mse = evaluate(
            model=model,
            tasks_loss_fn=tasks_loss_fn,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            in_domains=args.in_domains,
            log_images=log_images,
            mode="val",
            num_encoded_tokens=args.num_encoded_tokens,
            fp32_output_adapters=args.fp32_output_adapters.split("-"),
            dataset=args.dataset,
            print_freq=args.val_print_freq,
            total_num_tokens=args.total_num_tokens,
            MEAN=args.DATA_MEAN,
            STD=args.DATA_STD,
        )
        print(val_stats, val_stats.keys())
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        log_images = (
            args.log_wandb
            and args.log_images_wandb
            and (epoch % args.log_images_freq == 0)
        )
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            log_images=log_images,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            standardize_depth=args.standardize_depth,
            extra_norm_pix_loss=args.extra_norm_pix_loss,
            fp32_output_adapters=args.fp32_output_adapters.split("-"),
            dataset=args.dataset,
            print_freq=args.train_print_freq,
            total_num_tokens=args.total_num_tokens,
            is_mask_inputs=args.is_mask_inputs,
            use_opt_loss=args.use_opt_loss,
            use_cur_loss=args.use_cur_loss,
            use_tv_loss=args.use_tv_loss,
            loss_weights=args.loss_weights,
            MEAN=args.DATA_MEAN,
            STD=args.DATA_STD,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    loss_balancer=loss_balancer_without_ddp,
                    epoch=epoch,
                )
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            log_images = (
                args.log_wandb
                and args.log_images_wandb
                and (epoch % args.log_images_freq == 0)
            )
            val_stats, mae, mse = evaluate(
                model=model,
                tasks_loss_fn=tasks_loss_fn,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                in_domains=args.in_domains,
                log_images=log_images,
                mode="val",
                num_encoded_tokens=args.num_encoded_tokens,
                fp32_output_adapters=args.fp32_output_adapters.split("-"),
                dataset=args.dataset,
                print_freq=args.val_print_freq,
                total_num_tokens=args.total_num_tokens,
                MEAN=args.DATA_MEAN,
                STD=args.DATA_STD,
            )
            print("* MAE:{}, min MAE:{}".format(mae, min_val_loss))
            # if cur_mae < min_val_loss:
            if mae < min_val_loss:
                min_val_loss = mae
                best_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                    )
                print(f"New best val loss: {min_val_loss:.3f}")
            else:
                print(
                    f"Current best val loss: {min_val_loss:.3f} in epoch {best_epoch}"
                )

            log_stats = {
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"val/{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train/{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    tasks_loss_fn: Dict[str, torch.nn.Module],
    loss_balancer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = None,
    max_skip_norm: float = None,
    log_writer=None,
    log_images: bool = False,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_encoded_tokens: int = 196,
    in_domains: List[str] = [],
    loss_on_unmasked: bool = True,
    alphas: float = 1.0,
    sample_tasks_uniformly: bool = False,
    standardize_depth: bool = True,
    extra_norm_pix_loss: bool = False,
    fp32_output_adapters: List[str] = [],
    dataset: str = "FDST",
    print_freq: int = 10,
    total_num_tokens: int = 400,
    is_mask_inputs: bool = False,
    use_opt_loss=True,
    use_cur_loss=True,
    use_tv_loss=True,
    loss_weights: Dict[str, float] = {},
    MEAN: float = 0.0,
    STD: float = 1.0,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    tv_loss = TVLoss().to(device)
    fuse_loss = MaskedMSELoss().to(device)
    opt_loss = MaskedMSELoss().to(device)
    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
            if task != "name"
        }

        input_dict = {
            task: tensor for task, tensor in tasks_dict.items() if task in in_domains
        }
        if "mask" in tasks_dict.keys():
            mask = tasks_dict["mask"]
        B = input_dict["rgb"].shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            if is_mask_inputs:
                task_masks = {}
                task_masks["rgb"] = (
                    torch.zeros(B, total_num_tokens, device=device).detach().clone()
                )
                task_masks["density"] = (
                    torch.ones(B, total_num_tokens, device=device).detach().clone()
                )
                preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(
                    input_dict,
                    num_encoded_tokens=total_num_tokens,
                    mask_inputs=True,
                    task_masks=task_masks,
                )
            else:
                preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(
                    input_dict,
                    num_encoded_tokens=num_encoded_tokens,
                    alphas=alphas,
                    sample_tasks_uniformly=sample_tasks_uniformly,
                    fp32_output_adapters=fp32_output_adapters,
                    iter=epoch,
                )
            if "mask" in tasks_dict.keys():
                preds["density"][mask == 0] = -MEAN / STD
                preds_prev["density"][mask == 0] = -MEAN / STD
                preds_prev_warp[mask == 0] = -MEAN / STD
                pred_fuse[mask == 0] = -MEAN / STD

            if extra_norm_pix_loss:
                tasks_dict["norm_rgb"] = tasks_dict["rgb"]
                masks["norm_rgb"] = masks.get("rgb", None)

            task_losses = {}

            for task in preds:
                target = tasks_dict[task][:, :, -1]
                img_target = tasks_dict["rgb"][:, :, -1]

                with torch.cuda.amp.autocast(enabled=False):
                    if loss_on_unmasked:
                        if "cur" in loss_weights:
                            task_losses[task] = loss_weights["cur"] * (
                                tasks_loss_fn[task](preds[task].float(), target.float())
                            )
                        if "opt" in loss_weights:
                            task_losses["opt"] = loss_weights["opt"] * opt_loss(
                                img_warp.float(), img_target.float()
                            )
                        task_losses["fuse"] = loss_weights["fus"] * fuse_loss(
                            pred_fuse.float(), target.float()
                        )
                    else:
                        task_losses[task] = 10 * tasks_loss_fn[task](
                            preds[task].float(),
                            target.float(),
                            mask=masks.get(task, None),
                        )
                    if "tv" in loss_weights:
                        task_losses["tv"] = loss_weights["tv"] * (
                            tv_loss(pred_fuse.float()).mean()
                        )

            with torch.cuda.amp.autocast(enabled=False):
                weighted_task_losses = loss_balancer(task_losses)
                loss = sum(weighted_task_losses.values())

        if log_images and step == 0 and utils.is_main_process():
            # Just log images of first batch
            gt_images = {
                task: rearrange(v.detach().cpu().float(), "b c t h w -> b c (t h) w")
                for task, v in input_dict.items()
            }
            pred_images = {
                "cur": preds["density"].detach().cpu().float(),
                "fuse": pred_fuse.detach().cpu().float(),
                "opt": img_warp.detach().cpu().float(),
                "prev": preds_prev["density"].detach().cpu().float(),
                "prev_warp": preds_prev_warp.detach().cpu().float(),
            }

        with torch.cuda.amp.autocast(enabled=False):
            loss_value = sum(task_losses.values()).item()
            task_loss_values = {
                f"{task}_loss": l.item() for task, l in task_losses.items()
            }
            weighted_task_loss_values = {
                f"{task}_loss_weighted": l.item()
                for task, l in weighted_task_losses.items()
            }

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        with torch.cuda.amp.autocast(enabled=False):
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                skip_grad=max_skip_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    "loss": loss_value,
                    "lr": max_lr,
                    "weight_decay": weight_decay_value,
                    "grad_norm": grad_norm,
                }
            )
            log_writer.update(task_loss_values)
            log_writer.update(weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    if log_images and utils.is_main_process():
        prefix = "train/img"
        log_image_wandb(pred_images, gt_images, prefix=prefix, image_count=8)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        "[Epoch] " + k: meter.global_avg for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    tasks_loss_fn: Dict[str, torch.nn.Module],
    device: torch.device,
    epoch: int,
    log_images: bool = False,
    num_encoded_tokens: int = 196,
    in_domains: List[str] = [],
    loss_on_unmasked: bool = True,
    alphas: float = 1.0,
    fp32_output_adapters: List[str] = [],
    mode: str = "val",
    dataset: str = "FDST",
    print_freq: int = 50,
    total_num_tokens: int = 400,
    MEAN: float = 0.0,
    STD: float = 1.0,
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        f"{mode}_err",
        utils.SmoothedValue(window_size=1, fmt="{value:.4f} ({global_avg:.4f})"),
    )
    metric_logger.add_meter(
        f"{mode}_gt", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        f"{mode}_pred", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    if mode == "val":
        header = "(Eval) Epoch: [{}]".format(epoch)
    elif mode == "test":
        header = "(Test) Epoch: [{}]".format(epoch)
    else:
        raise ValueError(f"Invalid eval mode {mode}")
    mae = 0.0
    mse = 0.0
    pred_images = None
    gt_images = None
    maes = []
    for x in metric_logger.log_every(data_loader, print_freq, header):
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
            if task != "name"
        }

        input_dict = {
            task: tensor for task, tensor in tasks_dict.items() if task in in_domains
        }
        B = input_dict["rgb"].shape[0]
        task_masks = {}
        task_masks["rgb"] = torch.zeros(B, total_num_tokens, device=device)
        task_masks["density"] = torch.ones(B, total_num_tokens, device=device)
        if "mask" in tasks_dict.keys():
            mask = tasks_dict["mask"]
        with torch.cuda.amp.autocast():
            target = tasks_dict["density"][:, :, -1]
            input_dict["density"] = torch.rand_like(
                input_dict["density"], device=device
            )
            preds, masks, pred_fuse, img_warp, preds_prev, preds_prev_warp = model(
                input_dict,
                num_encoded_tokens=total_num_tokens,
                mask_inputs=True,
                task_masks=task_masks,
            )

            if "mask" in tasks_dict.keys():
                preds["density"][mask == 0] = -MEAN / STD
                preds_prev["density"][mask == 0] = -MEAN / STD
                preds_prev_warp[mask == 0] = -MEAN / STD
                pred_fuse[mask == 0] = -MEAN / STD
            input_dict["density"] = tasks_dict["density"]

        metrics = make_density_metrics(pred_fuse * STD + MEAN, target * STD + MEAN)
        pred_count = (pred_fuse * STD + MEAN).sum().cpu().item()
        gt_count = (target * STD + MEAN).sum().cpu().item()
        err = abs(pred_count - gt_count)
        mae += err
        mse += err**2
        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            gt_images = {
                task: rearrange(v.detach().cpu().float(), "b c t h w -> b c (t h) w")
                for task, v in input_dict.items()
            }
            pred_images = {
                "cur": preds["density"].detach().cpu().float(),
                "fuse": pred_fuse.detach().cpu().float(),
                "opt": img_warp.detach().cpu().float(),
                "prev": preds_prev["density"].detach().cpu().float(),
                "prev_warp": preds_prev_warp.detach().cpu().float(),
            }

        metric_logger.update(**{f"{mode}_err": err})
        metric_logger.update(**{f"{mode}_gt": gt_count})
        metric_logger.update(**{f"{mode}_pred": pred_count})
    mae = mae / len(data_loader)
    mse = (mse / len(data_loader)) ** 0.5
    print(f"* MAE {mae:.3f}, RMSE {mse:.3f}")
    if log_images and utils.is_main_process():
        prefix = "val/img" if mode == "val" else "test/img"
        log_image_wandb(pred_images, gt_images, prefix=prefix, image_count=8)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mae, mse


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)
