from glob import glob
from typing import List

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from engine import inference, train
from loader.image_loader import IdaoDataset, IdaoInferenceDataset
from loss import IdaoLoss, idao_metric
from model.backbone_multihead import TwoHeadModel
from utils.group_k_fold import group_k_fold

train_path: List[str] = glob("../data/track_1/idao_dataset/train/*/*.png")

fold_generator = group_k_fold(list_files=train_path, n_splits=2)

device = torch.device("cuda")

for index, (fold_train_path, fold_val_path) in enumerate(fold_generator):
    train_dataset = IdaoDataset(list_path=fold_train_path)
    val_dataset = IdaoDataset(list_path=fold_val_path)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8
    )

    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    resnet18 = models.resnet18(pretrained=True)
    backbone = nn.Sequential(nn.Sequential(*list(resnet18.children())[:-2]))
    model = TwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=5e-5)
    criterion = IdaoLoss(w_classification=1, w_regression=1)
    metric = idao_metric  # Callable
    scheduler = None

    dict_loader = {"train": train_dataloader, "validation": val_dataloader}
    num_epoch = 10

    train(
        model=model,
        optimizer=optimizer,
        num_epoch=10,
        dict_loader=dict_loader,
        criterion=criterion,
        metric=metric,
        scheduler=scheduler,
        device=device,
    )
