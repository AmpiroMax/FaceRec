import os
import typing as tp

import numpy as np
import torch
import torch.utils.tensorboard as tb
from src.data.dataset import CelebADataset
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

IMG_H = 128
IMG_W = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ANCHOR_CLASSES = int(os.environ["N_ANCHOR_CLASSES"])
MODEL_NAME = os.environ["MODEL_NAME"]

writer = tb.writer.SummaryWriter()


def train_epoch(
    model: nn.Module,
    train_dataset: CelebADataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.Optimizer,
    session_size: int,
    epoch_numer: int = 0,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    for step in tqdm(range(session_size)):
        model.zero_grad()
        data = train_dataset.get_batch(
            N_ANCHOR_CLASSES).view(-1, 3, IMG_W, IMG_H).to(DEVICE)

        embeddings = model(data).view(N_ANCHOR_CLASSES, 3, -1)

        loss = loss_func(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2]
        )

        dists_neg = torch.norm(embeddings[:, 0] - embeddings[:, 2], dim=1)
        dists_pos = torch.norm(embeddings[:, 0] - embeddings[:, 1], dim=1)
        accuracy = torch.mean(
            dists_neg > dists_pos + loss_func.margin,
            dtype=float
        )

        history["loss"] += [loss.item()]
        history["accuracy"] += [accuracy.item()]

        if tensorboard:
            writer.add_scalar(
                f"Training loss {MODEL_NAME}",
                history["loss"][-1],
                epoch_numer*session_size + step
            )
            writer.add_scalar(
                f"Training accuracy {MODEL_NAME}",
                history["accuracy"][-1],
                epoch_numer*session_size + step
            )
        loss.backward()

        if gradient_clip is not None:
            clip_grad_norm_(model.parameters(), gradient_clip)

        opt.step()

    return history


def train(
    model: nn.Module,
    train_dataset: CelebADataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.Optimizer,
    scheduler: optim.lr_scheduler.StepLR,
    session_size: int,
    epoch_num: int = 5,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None
) -> tp.Dict:
    history = {
        "loss": [],
        "accuracy": []
    }

    for epoch in range(epoch_num):
        epoch_hist = train_epoch(
            model,
            train_dataset,
            loss_func,
            opt,
            session_size,
            epoch,
            tensorboard,
            gradient_clip
        )

        history["loss"].extend(epoch_hist["loss"])
        history["accuracy"].extend(epoch_hist["accuracy"])

        scheduler.step()

    return history


def train_pipeline(
    model: nn.Module,
    trainable_params: tp.List,
    dataset: torch.utils.data.Dataset,
    lr: float = 1e-3,
    margin: float = 1.0,
    session_size: int = 100,
    epoch_num: int = 5,
    tensorboard: bool = False,
    gradient_clip: tp.Optional[int | None] = None
) -> tp.Dict:
    model.to(DEVICE)

    opt = optim.Adam(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.8, last_epoch=-1)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2.0)

    history = train(
        model=model,
        train_dataset=dataset,
        loss_func=triplet_loss,
        opt=opt,
        scheduler=scheduler,
        session_size=session_size,
        epoch_num=epoch_num,
        tensorboard=tensorboard,
        gradient_clip=gradient_clip
    )

    return history
