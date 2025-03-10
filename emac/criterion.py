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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        c = x.size()[1]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


@torch.cuda.amp.autocast(enabled=False)
class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(
            imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p
        )
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(
            x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p
        )
        return imgs

    def forward(self, input, target, mask=None):
        C = input.shape[1]
        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        loss = F.mse_loss(input, target, reduction="none")
        with torch.cuda.amp.autocast(enabled=False):
            if mask is not None:
                if mask.sum() == 0:
                    return torch.tensor(0).to(loss.device)

                mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
                mask = F.interpolate(
                    mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
                ).squeeze(1)
                loss_float = loss.float()
                loss_float = loss_float.mean(dim=1)
                loss_float = loss_float * mask
                # Compute mean per sample
                loss_float = loss_float.flatten(start_dim=1).sum(dim=1) / mask.flatten(
                    start_dim=1
                ).sum(dim=1)
                loss_float = loss_float.nanmean()  # Account for zero masks
                loss = loss_float
            else:
                loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
