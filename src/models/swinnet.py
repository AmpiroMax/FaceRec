""" Efficient module """

import typing as tp

from torch import nn
from torchvision.models import swin_t


class SwinNetModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
    ) -> None:
        super().__init__()
        self.model = swin_t(weights="IMAGENET1K_V1")
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=768, out_features=embedding_size)
        )

    def set_model_head_requires_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if name.split(".")[0] != "head":
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_model_to_finetune(self) -> None:
        self.model.eval()
        self.model.head.train()

    def get_trainable_params(self) -> tp.List:
        trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params += [param]
        return trainable_params

    def forward(self, img):
        return self.model(img)
