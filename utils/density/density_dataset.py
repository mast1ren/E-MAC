import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import glob
import h5py
import torchvision.transforms as T
import cv2
import random
import json
import scipy.io as sio


ImageFile.LOAD_TRUNCATED_IMAGES = True


class DensityDroneBird(Dataset):
    def __init__(
        self,
        data_root,
        # tasks,
        split="train",
        # variant='tiny',
        image_size=256,
        max_images=None,
        dens_norm=True,
        random_flip=True,
        return_unnorm=False,
        clip_size=2,
        stride=1,
        MEAN=None,
        STD=None,
    ):
        super(DensityDroneBird, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_images = max_images
        self.dens_norm = dens_norm
        self.random_flip = random_flip
        self.return_unnorm = return_unnorm
        self.clip_size = clip_size
        self.stride = stride
        self.MEAN = MEAN
        self.STD = STD
        with open(
            os.path.join(data_root, "{}.json".format(self.split)),
            "r",
        ) as f:
            self.image_ids = json.load(f)
        for img_idx in range(len(self.image_ids)):
            self.image_ids[img_idx] = os.path.join(data_root, self.image_ids[img_idx])
        print("imgs get")
        self.image_ids.sort()
        print("sorted")
        self.seqclips = []
        for img_idx in range(0, len(self.image_ids), self.stride):
            clip = [self.image_ids[img_idx]]
            cur_seq = os.path.basename(self.image_ids[img_idx])[3:6]
            pre_img = self.image_ids[img_idx]

            for i in range(1, self.clip_size):
                seq = os.path.basename(self.image_ids[max(0, img_idx - i)])[3:6]

                if seq == cur_seq:
                    pre_img = self.image_ids[max(0, img_idx - i)]

                clip.append(pre_img)

            self.seqclips.append(clip[::-1])
        print("cliped")
        if isinstance(self.max_images, int):
            start_idx = random.randint(0, max(len(self.seqclips) - self.max_images, 0))
            self.seqclips = self.seqclips[start_idx : start_idx + self.max_images]
        print(
            f"Initialized DensityDroneBird with {len(self.image_ids)} images in split {self.split}."
        )

    def __len__(self):
        return len(self.seqclips)
        # return len(self.image_ids)

    def __getitem__(self, index):
        transform_rgb = T.Compose(
            [
                T.Resize((1024, 2048)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
            # additional_targets={'image': 'image'},
        )
        transform_density = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize(FDST_MEAN, FDST_STD),
            ],
            # additional_targets={'denisty': 'mask'},
        )
        cur_density_path = (
            self.seqclips[index][0]
            .replace("images", "ground_truth")
            .replace("img", "fGT_img")
            .replace("jpg", "h5")
        )

        imgs = torch.stack(
            [
                transform_rgb(Image.open(img_path).convert("RGB"))
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )  # C, T, H, W
        c, t, h, w = imgs.shape
        cur_density = h5py.File(cur_density_path, "r")["density"][:]
        ori_shape = cur_density.shape
        dens = torch.stack(
            [
                transform_density(
                    (
                        cv2.resize(
                            h5py.File(
                                img_path.replace("images", "ground_truth")
                                .replace("img", "fGT_img")
                                .replace("jpg", "h5"),
                                "r",
                            )["density"][:],
                            (w, h),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        * (ori_shape[0] * ori_shape[1])
                        / (w * h)
                    ).astype("float32", copy=False)
                )
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )
        result = {}

        if self.image_size is not None:
            while True:
                x1, y1 = random_crop(imgs, cur_density, self.image_size)
                temp_dens = dens[
                    :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                ]
                # print(temp_dens.sum(dim=[0, 2, 3]))
                if (
                    torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0.5
                    or (
                        random.random() < 0.2
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0
                    )
                    or (
                        random.random() < 0.1
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) == 0
                    )
                    or self.split != "train"
                ):
                    dens = temp_dens
                    imgs = imgs[
                        :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                    ]
                    break
        else:
            imgs = imgs
            dens = dens

        if self.random_flip and random.random() < 0.4:
            imgs = torch.flip(imgs, dims=[3])
            dens = torch.flip(dens, dims=[3])

        if self.return_unnorm:
            result["unnorm_density"] = cur_density

        if self.dens_norm:
            dens = torch.stack(
                [
                    T.Normalize(self.MEAN, self.STD)(dens[:, t])
                    for t in range(dens.shape[1])
                ],
                dim=1,
            )
        result["rgb"] = imgs
        result["density"] = dens
        result["name"] = self.seqclips[index][0].split("/")[-1]

        return result


class DensityVSCrowd(Dataset):
    def __init__(
        self,
        data_root,
        # tasks,
        split="train",
        # variant='tiny',
        image_size=256,
        max_images=None,
        dens_norm=True,
        random_flip=True,
        return_unnorm=False,
        clip_size=2,
        stride=1,
        MEAN=None,
        STD=None,
    ):
        super(DensityVSCrowd, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_images = max_images
        self.dens_norm = dens_norm
        self.random_flip = random_flip
        self.return_unnorm = return_unnorm
        self.clip_size = clip_size
        self.stride = stride
        self.image_ids = glob.glob(
            os.path.join(data_root, "train_*" if split != "test" else "test_*", "*.jpg")
        )
        self.MEAN = MEAN
        self.STD = STD
        print("imgs get")
        self.image_ids.sort()
        print("sorted")
        self.seqclips = []
        for img_idx in range(0, len(self.image_ids), self.stride):
            clip = [self.image_ids[img_idx]]
            cur_seq = os.path.dirname(self.image_ids[img_idx]).split("/")[-1]
            pre_img = self.image_ids[img_idx]

            for i in range(1, self.clip_size):
                seq = os.path.dirname(self.image_ids[max(0, img_idx - i)]).split("/")[
                    -1
                ]

                if seq == cur_seq:
                    pre_img = self.image_ids[max(0, img_idx - i)]

                clip.append(pre_img)

            self.seqclips.append(clip[::-1])
        print("cliped")
        if self.split == "train":
            self.seqclips = self.seqclips[: -int(len(self.seqclips) * 0.1)]
        elif self.split == "val":
            self.seqclips = self.seqclips[-int(len(self.seqclips) * 0.1) :]
        if isinstance(self.max_images, int):
            start_idx = random.randint(0, max(len(self.seqclips) - self.max_images, 0))
            self.seqclips = self.seqclips[start_idx : start_idx + self.max_images]
        print(
            f"Initialized DensityVSCrowd with {len(self.seqclips)} images in split {self.split}."
        )

    def __len__(self):
        return len(self.seqclips)
        # return len(self.image_ids)

    def __getitem__(self, index):
        transform_rgb = T.Compose(
            [
                # T.Resize((1024, 2048)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )
        transform_density = T.Compose(
            [
                T.ToTensor(),
            ],
        )
        cur_density_path = self.seqclips[index][0].replace("jpg", "h5")

        imgs = torch.stack(
            [
                transform_rgb(Image.open(img_path).convert("RGB"))
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )  # C, T, H, W
        c, t, h, w = imgs.shape
        cur_density = h5py.File(cur_density_path, "r")["density"][:]
        ori_shape = cur_density.shape
        dens = torch.stack(
            [
                transform_density(
                    (
                        cv2.resize(
                            h5py.File(
                                img_path.replace("jpg", "h5"),
                                "r",
                            )[
                                "density"
                            ][:],
                            (w, h),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        * (ori_shape[0] * ori_shape[1])
                        / (w * h)
                    ).astype("float32", copy=False)
                )
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )
        result = {}

        if self.image_size is not None:
            while True:
                x1, y1 = random_crop(imgs, cur_density, self.image_size)
                temp_dens = dens[
                    :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                ]
                # print(temp_dens.sum(dim=[0, 2, 3]))
                if (
                    torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0.5
                    or (
                        random.random() < 0.2
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0
                    )
                    or (
                        random.random() < 0.1
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) == 0
                    )
                    or self.split != "train"
                ):
                    dens = temp_dens
                    imgs = imgs[
                        :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                    ]
                    break
        else:
            imgs = imgs
            dens = dens

        if self.random_flip and random.random() < 0.4:
            imgs = torch.flip(imgs, dims=[3])
            dens = torch.flip(dens, dims=[3])

        if self.return_unnorm:
            result["unnorm_density"] = cur_density

        if self.dens_norm:
            dens = torch.stack(
                [
                    T.Normalize(self.MEAN, self.STD)(dens[:, t])
                    for t in range(dens.shape[1])
                ],
                dim=1,
            )
        result["rgb"] = imgs
        result["density"] = dens
        result["name"] = (
            self.seqclips[index][0].split("/")[-2]
            + "_"
            + self.seqclips[index][0].split("/")[-1]
        )

        return result


class DensityFDST(Dataset):
    def __init__(
        self,
        data_root,
        tasks,
        split="train",
        variant="tiny",
        clip_size=2,
        image_size=256,
        max_images=None,
        dens_norm=True,
        random_flip=True,
        return_unnorm=False,
        stride=1,
        opticalflow=False,
        MEAN=None,
        STD=None,
    ):
        """
        Density dataloader.

        Args:
            data_root: Root of Density data directory
            tasks: List of tasks. Any of ['rgb', 'depth_euclidean', 'depth_zbuffer',
                'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal',
                'principal_curvature', 'reshading', 'mask_valid'].
            split: One of {'train', 'val', 'test'}
            variant: One of {'debug', 'tiny', 'medium', 'full', 'fullplus'}
            image_size: Target image size
            max_images: Optional subset selection
        """
        super(DensityFDST, self).__init__()
        self.data_root = data_root
        self.tasks = tasks
        self.split = split
        self.variant = variant
        self.clip_size = clip_size
        self.image_size = image_size
        self.max_images = max_images
        self.dens_norm = dens_norm
        self.random_flip = random_flip
        self.return_unnorm = return_unnorm
        self.stride = stride
        self.opticalflow = opticalflow
        self.MEAN = MEAN
        self.STD = STD
        self.image_ids = glob.glob(
            os.path.join(
                self.data_root,
                self.split + "_data" if self.split != "val" else "train_data",
                "*/*_resize.h5",
            )
        )
        print("imgs get")
        self.image_ids.sort()
        print("sorted")
        self.seqclips = []

        for img_idx in range(0, len(self.image_ids), self.stride):
            clip = [self.image_ids[img_idx]]
            cur_seq = os.path.dirname(self.image_ids[img_idx]).split("/")[-1]
            # cur_seq = os.path.basename(self.image_ids[img_idx])[-10:-7]
            pre_img = self.image_ids[img_idx]

            for i in range(1, self.clip_size):
                seq = os.path.dirname(self.image_ids[max(0, img_idx - i)]).split("/")[
                    -1
                ]

                if seq == cur_seq:
                    pre_img = self.image_ids[max(0, img_idx - i)]

                clip.append(pre_img)

            self.seqclips.append(clip[::-1])
        print("cliped")
        if self.split == "train":
            self.seqclips = self.seqclips[: -int(len(self.seqclips) * 0.1)]
        elif self.split == "val":
            self.seqclips = self.seqclips[-int(len(self.seqclips) * 0.1) :]
        # random.shuffle(self.seqclips)
        if isinstance(self.max_images, int):
            start_idx = random.randint(0, max(len(self.seqclips) - self.max_images, 0))
            self.seqclips = self.seqclips[start_idx : start_idx + self.max_images]

        print(
            f"Initialized DensityFDST with {len(self.image_ids)} images from variant {self.variant} in split {self.split}."
        )

    def __len__(self):
        # return len(self.image_ids)
        return len(self.seqclips)

    def __getitem__(self, index):
        transform_rgb = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
            # additional_targets={'image': 'image'},
        )
        transform_density = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize(FDST_MEAN, FDST_STD),
            ],
            # additional_targets={'denisty': 'mask'},
        )

        cur_density_path = self.seqclips[index]

        imgs = torch.stack(
            [
                transform_rgb(
                    Image.open(
                        img_path.replace("_resize", "").replace("h5", "jpg")
                    ).convert("RGB")
                )
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )  # C, T, H, W
        c, t, h, w = imgs.shape
        cur_density = h5py.File(cur_density_path[0], "r")["density"][:]
        ori_shape = cur_density.shape
        dens = torch.stack(
            [
                transform_density(
                    (
                        cv2.resize(
                            h5py.File(den_path, "r")["density"][:],
                            (w, h),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        * (ori_shape[0] * ori_shape[1])
                        / (w * h)
                    ).astype("float32", copy=False)
                )
                for den_path in self.seqclips[index]
            ],
            dim=1,
        )
        if self.opticalflow:
            optical_flow = h5py.File(
                cur_density_path[-1].replace("_resize", "_of2"), "r"
            )["opticalflow"][:]
            optical_flow = torch.from_numpy(optical_flow)

        result = {}

        if self.image_size is not None:
            while True:
                x1, y1 = random_crop(imgs, cur_density, self.image_size)
                temp_dens = dens[
                    :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                ]
                if (
                    torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0.5
                    or (
                        random.random() < 0.2
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0
                    )
                    or (
                        random.random() < 0.1
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) == 0
                    )
                    or self.split != "train"
                ):
                    dens = temp_dens
                    imgs = imgs[
                        :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                    ]
                    if self.opticalflow:
                        optical_flow = optical_flow[
                            :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                        ]
                    break
        else:
            imgs = imgs
            dens = dens

            if self.opticalflow:
                optical_flow = optical_flow

        if self.random_flip and random.random() < 0.4:
            imgs = torch.flip(imgs, dims=[3])
            dens = torch.flip(dens, dims=[3])
            if self.opticalflow:
                optical_flow = torch.flip(optical_flow, dims=[2])

        if self.return_unnorm:
            result["unnorm_density"] = cur_density

        if self.dens_norm:
            dens = torch.stack(
                [
                    T.Normalize(self.MEAN, self.STD)(dens[:, t])
                    for t in range(dens.shape[1])
                ],
                dim=1,
            )
        result["rgb"] = imgs
        result["density"] = dens
        result["name"] = self.seqclips[index][0].split("/")[-1]
        if self.opticalflow:
            result["opticalflow"] = optical_flow

        return result


class DensityMall(Dataset):
    def __init__(
        self,
        data_root,
        tasks,
        split="train",
        variant="tiny",
        image_size=256,
        max_images=None,
        dens_norm=True,
        random_flip=True,
        return_unnorm=False,
        clip_size=1,
        stride=1,
        MEAN=None,
        STD=None,
    ):
        super(DensityMall, self).__init__()
        self.data_root = data_root
        self.tasks = tasks
        self.split = split
        self.variant = variant
        self.image_size = image_size
        self.max_images = max_images
        self.dens_norm = dens_norm
        self.random_flip = random_flip
        self.return_unnorm = return_unnorm
        self.clip_size = clip_size
        self.stride = stride
        self.MEAN = MEAN
        self.STD = STD
        self.image_ids = glob.glob(
            os.path.join(
                self.data_root,
                "*/*.h5",
            )
        )
        print("imgs get")
        self.image_ids.sort()
        print("sorted")
        self.seqclips = []
        for img_idx in range(len(self.image_ids)):
            clip = [self.image_ids[img_idx]]
            clip.append(self.image_ids[max(img_idx - 1, 0)])
            self.seqclips.append(clip[::-1])
        print("cliped")
        if self.split == "train":
            tempseqclips = self.seqclips[:800]
            self.seqclips = tempseqclips
        else:
            self.seqclips = self.seqclips[800:]
        if isinstance(self.max_images, int):
            start_idx = random.randint(0, max(len(self.seqclips) - self.max_images, 0))
            self.seqclips = self.seqclips[start_idx : start_idx + self.max_images]
        print(
            f"Initialized DensityMall with {len(self.seqclips)} images from variant {self.variant} in split {self.split}."
        )

    def __len__(self):
        return len(self.seqclips)
        # return len(self.image_ids)

    def __getitem__(self, index):
        if self.split == "train":
            transform_rgb = T.Compose(
                [
                    # T.RandomAdjustSharpness(0.8, p=0.5),
                    # T.RandomAutocontrast(p=0.5),
                    # T.RandomEqualize(p=0.5),
                    T.Resize((448, 640)),
                    # T.Resize((1024, 1536)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ],
            )
        else:
            transform_rgb = T.Compose(
                [
                    T.Resize((448, 640)),
                    # T.Resize((1024, 1536)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ],
            )
        transform_density = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize(FDST_MEAN, FDST_STD),
            ],
        )
        cur_density_path = self.seqclips[index]
        imgs = torch.stack(
            [
                transform_rgb(
                    Image.open(img_path.replace(".h5", ".jpg")).convert("RGB")
                )
                for img_path in self.seqclips[index]
            ],
            dim=1,
        )  # C, T, H, W
        c, t, h, w = imgs.shape
        # print(c, t, h, w)
        cur_density = h5py.File(cur_density_path[0], "r")["density"][:]
        ori_shape = cur_density.shape
        dens = torch.stack(
            [
                transform_density(
                    (
                        cv2.resize(
                            h5py.File(den_path, "r")["density"][:],
                            (w, h),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        * (ori_shape[0] * ori_shape[1])
                        / (w * h)
                    ).astype("float32", copy=False)
                )
                for den_path in self.seqclips[index]
            ],
            dim=1,
        )
        mask = sio.loadmat(os.path.join(self.data_root, "perspective_roi.mat"))["roi"][
            "mask"
        ][0, 0]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = transform_density(mask)
        result = {}

        count_crop = 0
        if self.image_size is not None and self.split == "train":
            while True:
                x1, y1 = random_crop(imgs, cur_density, self.image_size)
                # x1, y1 = random_crop(cur_img, cur_density, self.image_size)
                temp_dens = dens[
                    :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                ]

                if (
                    torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0.5
                    or (
                        random.random() < 0.2
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) > 0
                    )
                    or (
                        random.random() < 0.1
                        and torch.mean(torch.sum(temp_dens, dim=[0, 2, 3])) == 0
                    )
                    or self.split != "train"
                ):
                    final_dens = temp_dens
                    final_imgs = imgs[
                        :, :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                    ]
                    final_mask = mask[
                        :, y1 : y1 + self.image_size, x1 : x1 + self.image_size
                    ]
                    break
        else:
            final_dens = dens
            final_imgs = imgs
            final_mask = mask

        # print('random density')
        if self.random_flip and random.random() < 0.4 and self.split == "train":
            final_imgs = torch.flip(final_imgs, dims=[3])
            final_dens = torch.flip(final_dens, dims=[3])
            final_mask = torch.flip(final_mask, dims=[2])

        if self.return_unnorm:
            result["unnorm_density"] = cur_density

        if self.dens_norm:
            final_dens = torch.stack(
                [
                    T.Normalize(self.MEAN, self.STD)(final_dens[:, t])
                    for t in range(final_dens.shape[1])
                ],
                dim=1,
            )
        result["rgb"] = final_imgs
        result["density"] = final_dens
        result["mask"] = final_mask
        result["name"] = os.path.basename(cur_density_path[0])

        return result


def random_crop(imgs, density, crop_size, optical=None):
    h, w = imgs.shape[-2:]
    th, tw = crop_size, crop_size
    if w == tw and h == th:
        return 0, 0

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return x1, y1


def buildDensityDataset(
    data_root,
    tasks,
    split="train",
    variant="tiny",
    image_size=256,
    max_images=None,
    dens_norm=True,
    random_flip=True,
    return_unnorm=False,
    dataset_name="FDST",
    clip_size=3,
    stride=2,
    opticalflow=False,
    MEAN=[0.0],
    STD=[1.0],
):
    if dataset_name == "FDST":
        return DensityFDST(
            data_root,
            variant=variant,
            split=split,
            tasks=tasks,
            image_size=image_size,
            max_images=max_images,
            dens_norm=dens_norm,
            random_flip=random_flip,
            return_unnorm=return_unnorm,
            clip_size=clip_size,
            stride=stride,
            opticalflow=opticalflow,
            MEAN=MEAN,
            STD=STD,
        )
    elif dataset_name == "Mall":
        return DensityMall(
            data_root,
            variant=variant,
            split=split,
            tasks=tasks,
            image_size=image_size,
            max_images=max_images,
            dens_norm=dens_norm,
            random_flip=random_flip,
            return_unnorm=return_unnorm,
            clip_size=clip_size,
            stride=stride,
            MEAN=MEAN,
            STD=STD,
        )
    elif dataset_name == "DroneBird":
        return DensityDroneBird(
            data_root,
            split=split,
            image_size=image_size,
            max_images=max_images,
            dens_norm=dens_norm,
            random_flip=random_flip,
            return_unnorm=return_unnorm,
            clip_size=clip_size,
            stride=stride,
            MEAN=MEAN,
            STD=STD,
        )
    elif dataset_name == "VSCrowd":
        return DensityVSCrowd(
            data_root,
            split=split,
            image_size=image_size,
            max_images=max_images,
            dens_norm=dens_norm,
            random_flip=random_flip,
            return_unnorm=return_unnorm,
            MEAN=MEAN,
            STD=STD,
        )
    else:
        raise NotImplementedError
