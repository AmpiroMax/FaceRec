import typing as tp
import torch

from src.data.dataset import CelebADataset
from torch import nn, optim
from tqdm.auto import tqdm
import torch.utils.tensorboard as tb


IMG_H = 128
IMG_W = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_SIZE = 300
N_ANCHOR_CLASSES = 10

BATCH_SIZE = 64
LR = 1e-3
MOMENTUM = 0.9

writer = tb.writer.SummaryWriter()


def train_epoch(
    model: nn.Module,
    train_dataset: CelebADataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.SGD,
    session_size: int,
    epoch_numer: int = 0,
    tensorboard: bool = False
) -> tp.Dict:
    history = {
        "loss": []
    }

    for step in tqdm(range(session_size)):
        model.zero_grad()
        data = train_dataset.get_batch(
            N_ANCHOR_CLASSES).view(-1, 3, IMG_W, IMG_H).to(DEVICE)

        embeddings = model(data).view(N_ANCHOR_CLASSES, 3, EMBEDDING_SIZE)

        loss = loss_func(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2]
        )

        history["loss"] += [loss.item()]
        if tensorboard:
            writer.add_scalar(
                "Training",
                history["loss"][-1],
                epoch_numer*session_size + step
            )
        loss.backward()
        opt.step()

    return history


def train(
    model: nn.Module,
    train_dataset: CelebADataset,
    loss_func: nn.TripletMarginLoss,
    opt: torch.optim.SGD,
    scheduler: optim.lr_scheduler.StepLR,
    session_size: int,
    epoch_num: int = 5,
    tensorboard: bool = False
) -> tp.Dict:
    history = {
        "loss": []
    }

    for epoch in range(epoch_num):
        epoch_hist = train_epoch(
            model,
            train_dataset,
            loss_func,
            opt,
            session_size,
            epoch,
            tensorboard
        )

        history["loss"].extend(epoch_hist["loss"])

        scheduler.step()

    return history
