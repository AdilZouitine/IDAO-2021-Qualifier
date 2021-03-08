from typing import NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metric import mean_absolute_error, roc_auc_score


class IdaoLoss(nn.Module):
    """[summary].

    Args:
        nn ([type]): [description]
    """

    def __init__(
        self, w_classification: float = 1, w_regression: float = 1
    ) -> NoReturn:
        """[summary].

        Args:
            w_classification (float, optional): [description]. Defaults to 1.
            w_regression (float, optional): [description]. Defaults to 1.
        """
        self.regression_loss = nn.L1Loss()
        self.classification_loss = nn.BCELoss()
        self.w_classification = w_classification
        self.w_regression = w_regression

    def __call__(
        self,
        predicted_class: torch.Tensor,
        true_class: torch.Tensor,
        predicted_angle: torch.Tensor,
        true_angle: torch.Tensor,
    ) -> torch.Tensor:
        classification_loss = self.classification_loss(
            predicted_class, true_class
        )  # noqa: E501
        regression_loss = self.regression_loss(predicted_angle, true_angle)
        loss = (
            self.w_classification * classification_loss
            + self.w_regression * regression_loss
        )
        return loss


def idao_metric(
    predicted_class: torch.Tensor,
    true_class: torch.Tensor,
    predicted_angle: torch.Tensor,
    true_angle: torch.Tensor,
) -> float:
    """[summary].

    Args:
        predicted_class (torch.Tensor): [description]
        true_class (torch.Tensor): [description]
        predicted_angle (torch.Tensor): [description]
        true_angle (torch.Tensor): [description]

    Returns:
        float: [description]
    """
    predicted_class = predicted_class.cpu().numpy()
    true_class = true_class.cpu().numpy()
    predicted_angle = predicted_angle.cpu().numpy()
    true_angle = true_angle.cpu().numpy()
    classif = roc_auc_score(true_class, predicted_class)
    regression = mean_absolute_error(true_angle, predicted_angle)
    return (classif - regression) * 100
