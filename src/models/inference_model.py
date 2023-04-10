

import albumentations as albu
import numpy as np
import torch
from torch import nn

MARGIN = 1


class FaceRecModel:
    def __init__(
        self,
        model: nn.Module,
        basic_transforms: albu.BaseCompose,
        augmentation_transforms: albu.BaseCompose
    ) -> None:
        self.model = model
        self.model.cpu().eval()
        self.basic_transforms = basic_transforms
        self.augmentation_transforms = augmentation_transforms
        self.emb_to_mean_num = 100
        self.margin = 0
        self.margin_std = 0
        self.owner_embeddings = None

    def learn_owner(self, img: np.ndarray) -> None:
        with torch.no_grad():
            mean_embedding = self.model(
                self.basic_transforms(image=img)["image"][None, ...]
            )

            for _ in range(self.emb_to_mean_num):
                mean_embedding += self.model(
                    self.augmentation_transforms(image=img)["image"][None, ...]
                )

            self.owner_embeddings = mean_embedding / self.emb_to_mean_num

            margins = []
            for _ in range(self.emb_to_mean_num):
                margins += [
                    torch.norm(
                        self.owner_embeddings - self.model(
                            self.augmentation_transforms(
                                image=img)["image"][None, ...]
                        )
                    ).detach().numpy()
                ]
            self.margin = np.mean(margins)
            self.margin_std = np.std(margins)

        print(self.margin + MARGIN + 3 * self.margin_std)

    def recognize_face(self, img: np.ndarray) -> int:
        with torch.no_grad():
            img = self.basic_transforms(image=img)["image"][None, ...]
            embedding = self.model(img)
        print(torch.norm(self.owner_embeddings - embedding))
        return torch.norm(self.owner_embeddings - embedding) < \
            self.margin + MARGIN + 3 * self.margin_std
