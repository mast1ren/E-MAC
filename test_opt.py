from utils.density.density_dataset import buildDensityDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    FDST_MEAN,
    FDST_STD,
    Mall_MEAN,
    Mall_STD,
    DB_MEAN,
    DB_STD,
    VSC_MEAN,
    VSC_STD,
)
from emac.emac import emac_base as e_mac
from emac.output_adapters import (
    SpatialOutputAdapter,
)
from emac.input_adapters import (
    PatchedInputAdapter,
)
import cv2
from einops import rearrange
import torchvision.transforms.functional as TF
import torch
import numpy as np
from functools import partial
import os
import warnings
import sys
from utils.pos_embed import interpolate_pos_embed_multimae
from emac.emac_utils import TransFuse

Normalization = {
    "FDST": {
        "MEAN": FDST_MEAN,
        "STD": FDST_STD,
    },
    "Mall": {
        "MEAN": Mall_MEAN,
        "STD": Mall_STD,
    },
    "DroneBird": {
        "MEAN": DB_MEAN,
        "STD": DB_STD,
    },
    "VSCrowd": {
        "MEAN": VSC_MEAN,
        "STD": VSC_STD,
    },
}

warnings.filterwarnings("ignore")


torch.set_grad_enabled(False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
DOMAIN_CONF = {
    "rgb": {
        "channels": 3,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=3, stride_level=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=3),
    },
    "density": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1, stride_level=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
    },
}
DOMAINS = ["rgb", "density"]


"""
inference setting
"""
image_height, image_width = 320, 320
data_path = "/path/to/data"
dataset = "dataset_name"
weight_path = "/path/to/weight"


input_adapters = {
    domain: DOMAIN_CONF[domain]["input_adapter"](
        stride_level=DOMAIN_CONF[domain]["stride_level"],
        patch_size_full=16,
        image_size=(image_height, image_width),
    )
    for domain in ["rgb", "density"]
}

output_adapters = {
    domain: DOMAIN_CONF[domain]["output_adapter"](
        stride_level=DOMAIN_CONF[domain]["stride_level"],
        patch_size_full=16,
        dim_tokens=256,
        depth=2,
        use_task_queries=True,
        task=domain,
        context_tasks=DOMAINS,
        image_size=(image_height, image_width),
    )
    for domain in ["density"]
}
fuse_module = TransFuse(
    stride_level=1,
    patch_size_full=16,
    image_size=(image_height, image_width),
    num_channels=1,
    dim_tokens_enc=768,
    dim_tokens=256,
    num_heads=8,
)

emac = e_mac(
    input_adapters=input_adapters,
    output_adapters=output_adapters,
    fuse_module=fuse_module,
    num_global_tokens=1,
)

idx = str(dataset)
ckpt_path = weight_path
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt_model = ckpt
print(ckpt_model.keys())

interpolate_pos_embed_multimae(emac, ckpt_model)

msg = emac.load_state_dict(ckpt_model, strict=False)
print(msg)

emac = emac.to(device).eval()


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def make_img_crop(img, img_height, img_width):
    if img.ndim == 3:
        c, h, w = img.shape
    elif img.ndim == 4:
        c, t, h, w = img.shape
    if h < img_height or w < img_width:
        raise ValueError("Image is too small for the desired crop size")
    hc, wc = int(h // img_height), int(w // img_width)

    if img.ndim == 3:
        img = TF.resize(img, (hc * img_height, wc * img_width))
        img = rearrange(img, "c (hc h) (wc w) -> (hc wc) c h w", hc=hc, wc=wc)
    elif img.ndim == 4:
        imgs = torch.zeros((hc * wc, c, t, img_height, img_width))
        for i in range(t):
            img_t = img[:, i]
            img_t = TF.resize(img_t, (hc * img_height, wc * img_width))
            img_t = rearrange(img_t, "c (hc h) (wc w) -> (hc wc) c h w", hc=hc, wc=wc)
            imgs[:, :, i] = img_t
        img = imgs
    return img, hc, wc


test_data = buildDensityDataset(
    data_path,
    tasks=["density"],
    split="test",
    variant="tiny",
    image_size=None,
    max_images=None,
    dens_norm=False,
    random_flip=False,
    dataset_name=dataset,
    clip_size=2,
    stride=1,
)

gt = []
pred = []

for i in range(len(test_data)):
    data = test_data[i]
    imgs_rgb = data["rgb"]
    den = data["density"]
    name = data["name"]
    den = den.numpy().astype(np.float32)
    gt_count = den[:, -1].sum()
    den = rearrange(den, "1 t h w -> t h w")
    fn, hd, wd = den.shape
    c, t, h, w = imgs_rgb.shape

    imgs_rgb, hc, wc = make_img_crop(imgs_rgb, image_height, image_width)

    n, c, t, h, w = imgs_rgb.shape
    dens_resize = np.zeros((t, hc * image_height, wc * image_width))
    for it in range(t):
        dens_resize[it] = (
            # den[i] = (
            cv2.resize(
                den[it],
                (wc * image_width, hc * image_height),
                interpolation=cv2.INTER_CUBIC,
            )
            * (hd * wd)
            / (wc * hc * image_width * image_height)
        )
    dens_resize = torch.from_numpy(dens_resize.astype(np.float32)[np.newaxis, ...])
    dens_resize_norm = (dens_resize - Normalization[dataset]["MEAN"]) / Normalization[
        dataset
    ]["STD"]
    dens_resize_norm, _, _ = make_img_crop(dens_resize_norm, image_height, image_width)
    den_pred_denorm = torch.zeros_like(dens_resize_norm).sum(dim=2)

    input_dict = {}
    input_dict["rgb"] = imgs_rgb
    input_dict["density"] = (
        torch.rand_like(input_dict["rgb"]).sum(dim=1, keepdim=True)
    )
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    num_encoded_tokens = int(image_height // 16 * image_width // 16)
    # num_encoded_tokens = 400  # the number of visible tokens
    # num_encoded_tokens = 1024
    task_masks = {}
    task_masks["rgb"] = torch.zeros(
        imgs_rgb.shape[0], num_encoded_tokens, device=device
    )
    task_masks["density"] = torch.ones(
        imgs_rgb.shape[0], num_encoded_tokens, device=device
    )
    (
        preds,
        masks,
        pred_fuse,
        img_warp,
        preds_prev,
        preds_prev_warp,
    ) = emac.forward(
        input_dict,
        mask_inputs=True,
        num_encoded_tokens=num_encoded_tokens,
        task_masks=task_masks,
    )
    den_pred = pred_fuse
    den_pred_denorm = (
        den_pred * Normalization[dataset]["STD"] + Normalization[dataset]["MEAN"]
    )

    den_pred_denorm = (
        rearrange(den_pred_denorm, "(hc wc) c h w -> c (hc h) (wc w)", hc=hc, wc=wc)
        .detach()
        .cpu()
    )

    pred_count = den_pred_denorm.sum()
    test_log = "{}: err: {:.4f}, gt_count: {:.4f}, pred_count: {:.4f}, name: {}".format(
        i,
        abs(gt_count - pred_count),
        gt_count,
        pred_count,
        name,
    )
    with open(
        os.path.join(os.path.dirname(ckpt_path), "{}_result.txt".format(idx)), "a"
    ) as f:
        f.write(test_log + "\n")
    print(test_log)
    gt.append(gt_count)
    pred.append(pred_count)

test_log = "MAE: {}, RMSE: {}".format(
    mean_absolute_error(gt, pred), mean_squared_error(gt, pred, squared=False)
)
print(test_log)
with open(
    os.path.join(os.path.dirname(ckpt_path), "{}_result.txt".format(idx)), "a"
) as f:
    f.write(test_log + "\n")
    f.write(ckpt_path)
print(ckpt_path)
