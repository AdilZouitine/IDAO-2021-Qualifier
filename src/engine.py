import math
from typing import Dict, Optional, Callable, NoReturn

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EarltStopping:
    """[summary]."""

    def __init__(self, tolerance: int, delta: float) -> NoReturn:
        """[summary].

        Args:
            tolerance (int): [description]
            delta (float): [description]
        """
        self.tolerance = tolerance
        self.delta = delta
        self.best_val_metric = -math.inf
        self.epoch_without_progress = 0

    def __call__(self, val_metric: float) -> bool:
        if val_metric >= (self.best_val_metric + self.delta):
            self.best_val_metric = val_metric
            self.epoch_without_progress = 0
        else:
            self.epoch_without_progress += 1

        if self.epoch_without_progress >= self.tolerance:
            return True
        return False


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epoch: int,
    dict_loader: Dict[str, DataLoader],
    criterion: nn.Module,
    metric: Callable,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cuda"),
) -> NoReturn:
    """[summary].

    Args:
        model (nn.Module): [description]
        optimizer (optim.Optimizer): [description]
        num_epoch (int): [description]
        dict_loader (Dict[str, DataLoader]): [description]
        criterion (nn.Module): [description]
        metric (Callable): [description]
        scheduler (Optional[optim.lr_scheduler._LRScheduler], optional):
            [description]. Defaults to None.
        device (torch.device, optional): [description].
            Defaults to torch.device("cuda").
    """
    for epoch in range(1, num_epoch + 1):
        print(f"EPOCH {epoch}")
        print("*" * 50)
        for phase in ["train", "validation"]:
            running_loss = 0.0
            list_target_class = []
            list_predicted_class = []
            list_target_angle = []
            list_predicted_angle = []

            if phase == "train":
                model.train()
            else:
                model.eval()
            for image, particule_class, particule_angle in tqdm(dict_loader[phase]):
                image = image.to(device)
                particule_class = particule_class.to(device)
                particule_angle = particule_angle.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    predicted_class, predicted_angle = model(image)

                    loss = criterion(
                        predicted_class=predicted_class,
                        true_class=particule_class,
                        predicted_angle=predicted_angle,
                        true_angle=particule_angle,
                    )

                    list_target_class.append(particule_class.detach().cpu())
                    list_predicted_class.append(predicted_class.detach().cpu())
                    list_target_angle.append(particule_angle.detach().cpu())
                    list_predicted_angle.append(predicted_angle.detach().cpu())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * image.size(0)

            epoch_loss = running_loss / len(dict_loader[phase].dataset)
            epoch_score, epoch_auc, epoch_mae = metric(
                predicted_class=list_predicted_class,
                true_class=list_target_class,
                predicted_kev=list_predicted_angle,
                true_kev=list_target_angle,
            )
            print(
                "{} Loss: {:.4f} Metric: {:.4f} -- Auc: {:.4f} -- Mae: {:.4f}".format(
                    phase, epoch_loss, epoch_score, epoch_auc, epoch_mae
                )  # noqa: E501
            )
        if scheduler is not None:
            scheduler.step()


def inference(
    model: nn.Module, loader: DataLoader, device: torch.device = torch.device("cuda")
):
    model.eval()
    dict_pred = {
        "id": [],
        "classification_predictions": [],
        "regression_predictions": [],
    }
    for image, image_name in loader:
        image = image.to(device)

        predicted_class, predicted_angle = model(image)
        predicted_class = predicted_class.cpu().numpy()
        predicted_angle = predicted_angle.cpu().numpy()
        dict_pred["id"].append(image_name)
        dict_pred["classification_predictions"].append(predicted_class)
        dict_pred["regression_predictions"].append(predicted_angle)
    return pd.DataFrame.from_dict(dict_pred)
