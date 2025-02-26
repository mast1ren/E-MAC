import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions.dirichlet import Dirichlet
import random
from utils.registry import register_model

from .models import pwc_dc_net

from .emac_utils import (
    Block,
    trunc_normal_,
    warp,
    denormalize,
)

__all__ = [
    "emac_base",
]


class EMac(nn.Module):
    """
    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def __init__(
        self,
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        fuse_module: nn.Module,
        num_global_tokens: int = 1,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        warmup: int = 40,
    ):
        super().__init__()
        self.warmup = warmup
        self.dim_tokens = dim_tokens
        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        fuse_module.init(dim_tokens_enc=dim_tokens)
        self.fuse = fuse_module

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        # Transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.encoder = nn.Sequential(
            *[
                Block(
                    dim=dim_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.pwc = pwc_dc_net()
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {"global_tokens"}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"input_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"output_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def sample_alphas(
        self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5
    ):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor(
            [list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:]
        )
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(
            valid_task_choices, 0, rand_per_sample_choice
        )
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks_with_ref(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
        ref_mask: torch.Tensor = None,
        dec_order: bool = None,
    ):
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device
        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()
        task_masks = []
        num_tokens_per_task = [
            task_tokens.shape[1] for task_tokens in input_tokens.values()
        ]

        # generate rgb mask ref density

        if dec_order == None:
            dec_order = random.random() <= 0.8

        noise_for_rgb = torch.nn.functional.unfold(
            rearrange(ref_mask, "b c h w->b c h w"), kernel_size=16, stride=16
        ).sum(dim=1)

        ids_arange_shuffle_rgb = torch.argsort(
            noise_for_rgb,
            dim=1,
            descending=dec_order,
        )
        mask_for_rgb = (
            torch.arange(num_tokens_per_task[0], device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        mask_for_rgb = torch.gather(mask_for_rgb, dim=1, index=ids_arange_shuffle_rgb)
        mask_for_rgb = torch.where(
            mask_for_rgb < samples_per_task[:, 0].unsqueeze(1), 0, 1
        )
        task_masks.append(mask_for_rgb)

        # generate density mask
        noise_for_density = torch.rand(B, num_tokens_per_task[1], device=device)
        ids_arange_shuffle_density = torch.argsort(noise_for_density, dim=1)
        mask_for_density = (
            torch.arange(num_tokens_per_task[1], device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        mask_for_density = torch.gather(
            mask_for_density, dim=1, index=ids_arange_shuffle_density
        )
        mask_for_density = torch.where(
            mask_for_density < samples_per_task[:, 1].unsqueeze(1), 0, 1
        )
        task_masks.append(mask_for_density)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {
            domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)
        }
        return task_masks, ids_keep, ids_restore, dec_order

    def generate_random_masks(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
        iter: int = 0,
    ):
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        samples_per_task = (task_sampling_dist * num_encoded_tokens).round().long()
        task_masks = []
        num_tokens_per_task = [
            task_tokens.shape[1] for task_tokens in input_tokens.values()
        ]
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            
            ids_arange_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)
            
        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {
            domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)
        }
        # print(task_masks['density'])

        return task_masks, ids_keep, ids_restore

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,  # TODO: Modify when adding non-2D tasks
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        mask_inputs: bool = True,
        task_masks: Dict[str, torch.Tensor] = None,
        num_encoded_tokens: int = 128,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
        fp32_output_adapters: List[str] = [],
        iter: int = 0,
        train: bool = True,
    ):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.

        :param x: Input tensor or dictionary of tensors
        :param mask_inputs: Set to True to enable random masking of input patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens to randomly select for encoder.
            Only used if mask_inputs is True.
        :param alphas: Dirichlet distribution parameter alpha for task sampling.
            Higher alpha = harder, less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True if tasks should be uniformly presampled,
            before Dirichlet sampling decides share of masked tokens between them.
        :param fp32_output_adapters: List of task identifiers to force output adapters to
            run with mixed precision turned off for stability reasons.
        """

        # Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {"rgb": x} if isinstance(x, torch.Tensor) else x

        # Need image size for tokens->image reconstruction
        # We assume that at least one of rgb or semseg is given as input before masking
        if "rgb" in x:
            # if train:
            B, C, T, H, W = x["rgb"].shape
        # else:
        # B, C, H, W = x['rgb'].shape
        elif "semseg" in x:
            B, H, W = x["semseg"].shape
            H *= self.input_adapters["semseg"].stride_level
            W *= self.input_adapters["semseg"].stride_level
        else:
            B, C, H, W = list(x.values())[
                0
            ].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor[:, :, -1])
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }
        input_info = self.generate_input_info(
            input_task_tokens=input_task_tokens, image_size=(H, W)
        )
        input_task_tokens_prev = {
            domain: self.input_adapters[domain](tensor[:, :, 0])
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }
        input_info_prev = self.generate_input_info(
            input_task_tokens=input_task_tokens_prev, image_size=(H, W)
        )
        # print(input_info)
        # Select random subset of tokens from the chosen input tasks and concatenate them

        if mask_inputs:
            num_encoded_tokens = (
                num_encoded_tokens
                if num_encoded_tokens is not None
                else self.num_encoded_tokens
            )
        else:
            num_encoded_tokens = sum(
                [tensor.shape[1] for tensor in input_task_tokens.values()]
            )
        # Generating masks
        if task_masks is None:
            # task_masks, ids_keep, ids_restore = self.generate_random_masks(
            #     input_task_tokens,
            #     num_encoded_tokens,
            #     alphas=alphas,
            #     sample_tasks_uniformly=sample_tasks_uniformly,
            #     iter=iter,
            # )
            # (
            #     task_masks,
            #     ids_keep,
            #     ids_restore,
            # ) = self.generate_random_masks(
            #     input_task_tokens,
            #     num_encoded_tokens,
            #     alphas=alphas,
            #     sample_tasks_uniformly=sample_tasks_uniformly
            # )
            # (
            #     task_masks_prev,
            #     ids_keep_prev,
            #     ids_restore_prev,
            # ) = self.generate_random_masks(
            #     input_task_tokens_prev,
            #     num_encoded_tokens,
            #     alphas=alphas,
            #     sample_tasks_uniformly=sample_tasks_uniformly
            #     # dec_order=dec_order,
            # )
            (
                task_masks,
                ids_keep,
                ids_restore,
                dec_order,
            ) = self.generate_random_masks_with_ref(
                input_task_tokens,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
                ref_mask=x["density"][:, :, -1],
            )
            (
                task_masks_prev,
                ids_keep_prev,
                ids_restore_prev,
                _,
            ) = self.generate_random_masks_with_ref(
                input_task_tokens_prev,
                num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
                ref_mask=x["density"][:, :, 0],
                # dec_order=dec_order,
            )
        else:
            mask_all = torch.cat(
                [task_masks[task] for task in input_task_tokens.keys()], dim=1
            )
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :num_encoded_tokens]
            ids_restore_prev = ids_restore
            ids_keep_prev = ids_keep
            # print(num_encoded_tokens)
        # if train:
        input_tokens = torch.cat(
            [
                task_tokens
                for task, task_tokens in input_task_tokens.items()
                # if task != 'rgb'
            ],
            dim=1,
        )

        input_tokens = torch.gather(
            input_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]),
        )

        input_tokens_prev = torch.cat(
            [task_tokens for task, task_tokens in input_task_tokens_prev.items()], dim=1
        )
        input_tokens_prev = torch.gather(
            input_tokens_prev,
            dim=1,
            index=ids_keep_prev.unsqueeze(-1).repeat(1, 1, input_tokens_prev.shape[2]),
        )
        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, "() n d -> b n d", b=B)
        global_tokens_prev = repeat(self.global_tokens, "() n d -> b n d", b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)
        input_tokens_prev = torch.cat([input_tokens_prev, global_tokens_prev], dim=1)

        encoder_tokens = self.encoder(input_tokens)
        encoder_tokens_prev = self.encoder(input_tokens_prev)

        # Output decoders
        if self.output_adapters is None:
            return encoder_tokens, task_masks

        # Decode tokens for each task using task-specific output adapters
        # with torch.cuda.amp.autocast(enabled=False):
        preds = self.output_adapters["density"](
            encoder_tokens=encoder_tokens,
            # temporal_tokens=adj_tokens,
            input_info=input_info,
            ids_keep=ids_keep,
            ids_restore=ids_restore,
        )
        preds = {"density": preds}
        preds_prev = self.output_adapters["density"](
            encoder_tokens=encoder_tokens_prev,
            input_info=input_info_prev,
            ids_keep=ids_keep_prev,
            ids_restore=ids_restore_prev,
        )
        preds_prev = {"density": preds_prev}

        # Force running selected output adapters in fp32 mode
        with torch.cuda.amp.autocast(enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )
                preds_prev[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens_prev.float(),
                    input_info=input_info_prev,
                    ids_keep=ids_keep_prev,
                    ids_restore=ids_restore_prev,
                )

        permute = [2, 1, 0]
        img_all = torch.cat(
            [
                denormalize(x["rgb"][:, :, -1])[:, permute] / 255.0,
                denormalize(x["rgb"][:, :, 0])[:, permute] / 255.0,
            ],
            dim=1,
        ).to(x["rgb"].device)
        flo = self.pwc(img_all)
        flo_shape = flo.shape[-2:]
        flo = (
            (
                nn.functional.interpolate(
                    flo, (H, W), mode="bilinear", align_corners=False
                )
            )
            * (flo_shape[0] * flo_shape[1])
            / (H * W)
        )
        img_warp = warp(x["rgb"][:, permute, 0], flo)[:, permute]

        preds_prev_warp = warp(preds_prev["density"], flo.detach())
        pred_fuse = self.fuse_dense(preds_prev_warp, preds["density"])

        return preds, task_masks, pred_fuse, img_warp, preds_prev, preds_prev_warp

    def fuse_dense(self, pred_prev_warp, pred_cur):
        pred_prev_warp_token = self.input_adapters["density"](pred_prev_warp)
        pred_cur_token = self.input_adapters["density"](pred_cur)
        return pred_cur + self.fuse(pred_prev_warp_token, pred_cur_token)


@register_model
def emac_base(
    input_adapters: Dict[str, nn.Module],
    output_adapters: Optional[Dict[str, nn.Module]],
    fuse_module: nn.Module,
    **kwargs,
):
    model = EMac(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        fuse_module=fuse_module,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
