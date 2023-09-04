import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os
import pickle
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as F

from datadings.reader import MsgpackReader

class SquarePad:
    """
    Pads the image to right side with given backgroud pixel values
    """

    pad_value: int = 255

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int(max_wh - w)
        vp = int(max_wh - h)
        padding = (0, 0, hp, vp)
        return F.pad(image, padding, self.pad_value, "constant")

@Registers.datasets.register_with_name("custom_single")
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(
            image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name("custom_aligned")
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/B")
        )
        image_paths_cond = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, f"{stage}/A")
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(
            image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal
        )
        self.imgs_cond = ImagePathDataset(
            image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal
        )

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name("custom_colorization_LAB")
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1.0, 1.0)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name("custom_colorization_RGB")
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        cond_image = image.convert("L")
        cond_image = cond_image.convert("RGB")

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.0
            image.clamp_(-1.0, 1.0)
            cond_image = (cond_image - 0.5) * 2.0
            cond_image.clamp_(-1.0, 1.0)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name("custom_inpainting")
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage="train"):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(
            os.path.join(dataset_config.dataset_path, stage)
        )
        self.flip = dataset_config.flip if stage == "train" else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.0
            image.clamp_(-1.0, 1.0)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[
            :,
            mask_pos_x : mask_pos_x + mask_height,
            mask_pos_y : mask_pos_y + mask_width,
        ] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name("shabby_pages_fixed_resolution")
class ShabbyPagesFixedResolution(data.Dataset):
    def __init__(self, dataset_config, stage="train"):
        self.dataset_path = dataset_config.dataset_path
        self.image_size = dataset_config.image_size
        self.stage = stage
        self.stage = "validation" if self.stage == "val" else self.stage

        if self.stage == "train":
            self.tfs = transforms.Compose(
                [
                    SquarePad(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        else:
            self.tfs = transforms.Compose(
                [
                    SquarePad(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )

            self.gray_tfs = transforms.Compose(
                [
                    SquarePad(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        self.data_reader = MsgpackReader(
            Path(self.dataset_path) / self.stage / "train_512x512.msgpack"
        )

    def __getitem__(self, index):
        import io

        sample = pickle.loads(self.data_reader[index]["data"])
        cond_image = Image.open(io.BytesIO(sample["image"]))
        gt_image = Image.open(io.BytesIO(sample["gt_image"]))

        cond_image = self.tfs(cond_image)
        gt_image = self.tfs(gt_image)

        # apply data augmentation
        if self.stage == "train":
            # random horizontal flipping
            if random.random() > 0.5:
                gt_image = TF.hflip(gt_image)
                cond_image = TF.hflip(cond_image)

            # random vertical flipping
            if random.random() > 0.5:
                gt_image = TF.vflip(gt_image)
                cond_image = TF.vflip(cond_image)

        # print("stage", self.stage)
        # print(cond_image.shape, gt_image.shape)
        # import matplotlib.pyplot as plt

        # plt.imshow(cond_image.permute(1,2,0))
        # plt.show()

        # plt.imshow(gt_image.permute(1, 2, 0))
        # plt.show()
        # print(cond_image.min(), cond_image.max(), cond_image.mean(), cond_image.std())
        # print(gt_image.min(), gt_image.max(), gt_image.mean(), gt_image.std())

        return (gt_image, sample["gt_image_file_path"]), (
            cond_image,
            sample["image_file_path"],
        )

    def __len__(self):
        return len(self.data_reader)


@Registers.datasets.register_with_name("shabby_pages")
class ShabbyPages(data.Dataset):
    def __init__(self, dataset_config, stage="train"):
        self.dataset_path = dataset_config.dataset_path
        self.image_size = dataset_config.image_size
        self.stage = stage
        self.stage = "validation" if self.stage == "val" else self.stage

        if self.stage == "train":
            self.tfs = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        else:
            self.tfs = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )

            self.gray_tfs = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        self.data_reader = MsgpackReader(
            Path(self.dataset_path) / self.stage / "512x512.msgpack"
        )

    def __getitem__(self, index):
        import io
        from skimage.filters import threshold_sauvola

        sample = pickle.loads(self.data_reader[index]["data"])
        cond_image = Image.open(io.BytesIO(sample["image"]))
        gt_image = Image.open(io.BytesIO(sample["gt_image"]))

        # binarize gt_images
        # gt_image = np.array(gt_image.convert("L"))
        # window_size = 25
        # thresh_sauvola = threshold_sauvola(gt_image, window_size=window_size)
        # gt_image = gt_image > thresh_sauvola
        # gt_image = Image.fromarray(gt_image).convert("RGB")

        cond_image = self.tfs(cond_image)
        gt_image = self.gray_tfs(gt_image)

        # apply data augmentation
        if self.stage == "train":
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                gt_image, output_size=(self.image_size, self.image_size)
            )
            i, j, h, w = transforms.RandomCrop.get_params(
                cond_image, output_size=(self.image_size, self.image_size)
            )
            cond_image = TF.crop(cond_image, i, j, h, w)
            gt_image = TF.crop(gt_image, i, j, h, w)

            # random horizontal flipping
            if random.random() > 0.5:
                gt_image = TF.hflip(gt_image)
                cond_image = TF.hflip(cond_image)

            # random vertical flipping
            if random.random() > 0.5:
                gt_image = TF.vflip(gt_image)
                cond_image = TF.vflip(cond_image)
            # print(cond_image.shape, gt_image.shape)

        # print("stage", self.stage)
        # print(cond_image.shape, gt_image.shape)
        # import matplotlib.pyplot as plt

        # plt.imshow(cond_image.permute(1,2,0))
        # plt.show()

        # plt.imshow(gt_image.permute(1, 2, 0))
        # plt.show()
        # print(cond_image.min(), cond_image.max(), cond_image.mean(), cond_image.std())
        # print(gt_image.min(), gt_image.max(), gt_image.mean(), gt_image.std())

        return (gt_image, sample["gt_image_file_path"]), (
            cond_image,
            sample["image_file_path"],
        )

    def __len__(self):
        return len(self.data_reader)
