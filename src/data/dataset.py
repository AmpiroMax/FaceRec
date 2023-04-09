""" Dataset module """

import typing as tp

import albumentations as albu
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset
from tqdm.auto import tqdm


CELEBA_DATA_PATH = "../data/raw/CelebA/img_align_celeba/"


class CelebADataset(Dataset):
    def __init__(
        self,
        basic_transforms: albu.BaseCompose,
        data_path: str = CELEBA_DATA_PATH,
        data_type: str = "train",
        augmentation_transforms: tp.Optional[albu.BaseCompose] = None
    ) -> None:
        super().__init__()
        self.data_type = data_type
        self.data_path = data_path

        self.basic_transforms = basic_transforms
        self.augmentation_transforms = augmentation_transforms

        self.label2images_names = dict()
        self.label2count = dict()
        self.image_names = []
        self.labels = []

        self._initialize()
        self.num_of_classes = len(self.label2count.keys())

    def get_batch(
        self,
        n_way: int
    ) -> torch.Tensor:
        image_names = self._get_sample(n_way)

        images = torch.cat(
            [
                torch.cat(
                    [
                        self._get_image(anchor)[None, ...],
                        self._get_image(anchor)[None, ...],
                        self._get_image(negative)[None, ...],
                    ],
                    dim=0
                )[None, ...]
                for anchor, negative in image_names
            ],
            dim=0
        )

        return images

    def _initialize(self) -> None:
        number_of_images = 0
        with open(self.data_path + "labels.txt", "r") as f:
            number_of_images = sum(1 for _ in f)

        with open(self.data_path + "labels.txt", "r") as file:
            for line in tqdm(file, desc="Reading images", total=number_of_images):
                img_name, img_label = line.split()
                img_label = int(img_label)

                if img_label not in self.label2images_names:
                    self.label2images_names[img_label] = [img_name]
                    self.label2count[img_label] = 1
                else:
                    self.label2images_names[img_label] += [img_name]
                    self.label2count[img_label] += 1

        for label, values in self.label2images_names.items():
            for img_name in values:
                self.image_names += [img_name]
                self.labels += [label]

    def _get_sample(
        self,
        n_anchor_classes: int
    ) -> np.ndarray:
        random_classes = np.random.choice(
            list(self.label2images_names.keys()),
            size=n_anchor_classes * 2,
            replace=False
        )

        images_names = np.array([
            np.random.choice(
                self.label2images_names[random_classe],
                size=1
            ).tolist() for random_classe in random_classes
        ]).reshape(n_anchor_classes, 2).tolist()

        return images_names

    def _get_image(
        self,
        image_name: str
    ) -> torch.Tensor:
        img = cv2.imread(self.data_path + "images/" + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentation_transforms is not None:
            img = self.augmentation_transforms(image=img)
        else:
            img = self.basic_transforms(image=img)

        return img["image"]

    def __getitem__(self, index: int) -> tp.Tuple:
        img = self._get_image(
            image_name=self.image_names[index]
        )

        return (img, self.labels[index])

    def __len__(self):
        return len(self.image_names)
