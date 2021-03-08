from typing import Tuple

import torch
import torch.nn as nn


class TwoHeadModel(nn.Module):
    def __init__(self, backbone_model: nn.Module, embedding_size: int):
        super().__init__(self)
        self.backbone_model = backbone_model
        self.emb_linear = nn.Linear(in_features=embedding_size, out_features=256)
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.emb_2 = nn.Linear(in_features=256, out_features=256)
        self.classif_head = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.regression_head = nn.Linear(in_features=256, out_features=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone_model(x)
        x = self.activation(x)
        x = self.emb_linear(x)
        x = self.activation(x)
        x = self.emb_2(x)
        x = self.activation(x)

        classif_logit = self.classif_head(x)
        classif_proba = self.sigmoid(classif_logit)

        regression = self.regression_head(x)

        return classif_proba, regression
