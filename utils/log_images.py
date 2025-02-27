# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb

import utils

def inv_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalization that was done during pre-processing"""
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    return inv_normalize(tensor)


@torch.no_grad()
def log_image_wandb(
    preds: Dict[str, torch.Tensor],
    gts: Dict[str, torch.Tensor],
    image_count=8,
    prefix="",
):
    pred_tasks = list(preds.keys())
    gt_tasks = list(gts.keys())
    if 'mask_valid' in gt_tasks:
        gt_tasks.remove('mask_valid')

    image_count = min(len(preds[pred_tasks[0]]), image_count)

    all_images = {}

    for i in range(image_count):
        # Log GTs
        for task in gt_tasks:
            gt_img = gts[task][i]
            if (
                'rgb' in task
                or task == 'rgb_cur'
                or task == 'rgb_prev0'
                or task == 'rgb'
            ):
                gt_img = inv_norm(gt_img)
            if gt_img.shape[0] == 1:
                gt_img = gt_img[0]
            elif gt_img.shape[0] == 2:
                gt_img = F.pad(gt_img, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0)

            gt_img = wandb.Image(gt_img, caption=f'GT #{i}')
            key = f'{prefix}_gt_{task}'
            if key not in all_images:
                all_images[key] = [gt_img]
            else:
                all_images[key].append(gt_img)

        # Log preds
        for task in pred_tasks:
            pred_img = preds[task][i]
            if task == 'rgb':
                pred_img = inv_norm(pred_img)
            if pred_img.shape[0] == 1:
                pred_img = pred_img[0]
            elif pred_img.shape[0] == 2:
                pred_img = F.pad(
                    pred_img, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0
                )

            pred_img = wandb.Image(pred_img, caption=f'Pred #{i}')
            key = f'{prefix}_pred_{task}'
            if key not in all_images:
                all_images[key] = [pred_img]
            else:
                all_images[key].append(pred_img)

    wandb.log(all_images, commit=False)
