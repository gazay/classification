from collections import OrderedDict
from typing import List

import cv2
import albumentations as A
from albumentations.pytorch import ToTensor

import torchvision

from catalyst.dl.experiments import ConfigExperiment
from .dataset import AlbuDataset

# ---- Augmentations ----
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
IMG_SIZE = 224


def train_transforms(image_size=224):
    transforms = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.RandomRotate90(),
        A.JpegCompression(quality_lower=50, p=0.3),
        A.Normalize(), ToTensor()
    ])
    return transforms


def infer_transform(image_size=224):
    transforms = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(), ToTensor()
    ])
    return transforms


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        if mode == "train":
            transform_fn = train_transforms(image_size=IMG_SIZE)
        elif mode in ["valid", "infer"]:
            transform_fn = infer_transform(image_size=IMG_SIZE)
        else:
            raise NotImplementedError

        return transform_fn

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = AlbuDataset(
            dataset=torchvision.datasets.CIFAR100(
                root='./data',
                train=True,
                download=True
            ),
            transforms=Experiment.get_transforms(stage, "train")
        )
        testset = AlbuDataset(
            dataset=torchvision.datasets.CIFAR100(
                root='./data',
                train=False,
                download=True
            ),
            transforms=Experiment.get_transforms(stage, "valid")
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets