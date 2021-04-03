from typing import NoReturn, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoHeadModel(nn.Module):
    def __init__(
        self, backbone_model: nn.Module, embedding_size: int
    ) -> NoReturn:  # noqa: E501
        super().__init__()
        self.backbone_model = backbone_model
        self.emb_linear = nn.Linear(in_features=embedding_size, out_features=256)
        self.flatten = nn.Flatten()
        self.final_pooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.emb_2 = nn.Linear(in_features=256, out_features=256)
        self.classif_head = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.regression_head = nn.Linear(in_features=256, out_features=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone_model(x)
        x = self.final_pooling(x)
        x = self.flatten(x)
        x = self.activation(x)
        x = self.emb_linear(x)
        x = self.activation(x)
        x = self.emb_2(x)
        x = self.activation(x)

        classif_logit = self.classif_head(x)
        classif_proba = self.sigmoid(classif_logit)

        regression = self.regression_head(x)

        return classif_proba, regression


class IndependantTwoHeadModel(nn.Module):
    def __init__(
        self, backbone_model: nn.Module, embedding_size: int
    ) -> NoReturn:  # noqa: E501
        super().__init__()
        self.backbone_model_classif = backbone_model
        self.backbone_model_regression = deepcopy(backbone_model)

        self.emb_linear_regression = nn.Linear(
            in_features=embedding_size, out_features=256
        )
        self.emb_linear_classif = nn.Linear(
            in_features=embedding_size, out_features=256
        )
        self.flatten = nn.Flatten()
        self.final_pooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.emb_2 = nn.Linear(in_features=256, out_features=256)
        self.classif_head = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.regression_head = nn.Linear(in_features=256, out_features=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        classif = self.backbone_model_classif(x)
        classif = self.final_pooling(classif)
        classif = self.flatten(classif)
        classif = self.activation(classif)
        classif = self.emb_linear_classif(classif)
        classif = self.activation(classif)
        classif = self.classif_head(classif)
        classif = self.sigmoid(classif)

        regression = self.backbone_model_regression(x)
        regression = self.final_pooling(regression)
        regression = self.flatten(regression)
        regression = self.activation(regression)
        regression = self.emb_linear_regression(regression)
        regression = self.activation(regression)
        regression = self.regression_head(regression)

        return classif, regression


class RegressionModel(nn.Module):
    def __init__(self, backbone_model: nn.Module, embedding_size: int):
        super().__init__()
        self.backbone_model_regression = backbone_model
        self.flatten = nn.Flatten()
        self.final_pooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.emb_linear_regression = nn.Linear(
            in_features=embedding_size, out_features=256
        )
        self.regression_head = nn.Linear(in_features=256, out_features=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        regression = self.backbone_model_regression(x)
        regression = self.final_pooling(regression)
        regression = self.flatten(regression)
        regression = self.activation(regression)
        regression = self.emb_linear_regression(regression)
        regression = self.activation(regression)
        regression = self.regression_head(regression)
        regression = F.softmax(regression, dim=1)
        return regression


def wrapper_regression(regression_value: torch.Tensor) -> torch.Tensor:
    """
    Convert energy to label.
    Args:
        regression_value (torch.Tensor): regression value
    Returns:
        torch.Tensor: output a class code
    """
    for index, energy in enumerate([1, 3, 6, 10, 20, 30]):
        regression_value = torch.where(
            regression_value == energy, index, regression_value
        )

    return regression_value


def pred_to_regression(
    regression_predicted: torch.Tensor, argmax: bool = True
) -> torch.Tensor:
    """
    Convert predicted class to predicted energy.

    Args:
        regression_predicted (torch.Tensor): [description]

    Returns:
        torch.Tensor: output an energy
    """
    if argmax:
        regression_predicted = torch.argmax(regression_predicted, dim=1)
    for index, energy in zip(range(5, -1, -1), reversed([1, 3, 6, 10, 20, 30])):

        regression_predicted = torch.where(
            regression_predicted == index, energy, regression_predicted
        )  # noqa: E501

    return regression_predicted
