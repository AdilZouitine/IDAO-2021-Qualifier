from glob import glob
from typing import List
from functools import partial

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from engine import inference, train
from loader.image_loader import IdaoDataset, IdaoInferenceDataset
from loss import IdaoLoss, idao_metric
from model.backbone_multihead import TwoHeadModel, IndependantTwoHeadModel
from submission_maker import SubmissionMaker
from utils.logo import logo


def main(mode="train", save_train=None, trained_model_path=None):
    device = torch.device("cuda")

    if mode == "train":
        train_path: List[str] = glob("data/idao_dataset/train/*/*.png")

        fold_generator = logo(list_files=train_path)

        for index, (fold_train_path, fold_val_path) in enumerate(fold_generator):
            print(f"FOLD {index}")
            print("#" * 150)
            train_dataset = IdaoDataset(list_path=fold_train_path)
            val_dataset = IdaoDataset(list_path=fold_val_path)

            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=38, shuffle=True, num_workers=8
            )

            val_dataloader = DataLoader(
                dataset=val_dataset, batch_size=38, shuffle=False, num_workers=8
            )

            resnet18 = models.resnet18(pretrained=True)
            backbone = nn.Sequential(nn.Sequential(*list(resnet18.children())[:-2]))
            model = TwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)
            # model = IndependantTwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)
            model.to(device)
            optimizer = optim.Adam(params=model.parameters(), lr=5e-5)
            criterion = IdaoLoss(w_classification=1, w_regression=1)
            metric = partial(idao_metric, do_round_kev=True)  # Callable
            scheduler = None

            dict_loader = {"train": train_dataloader, "validation": val_dataloader}

            num_epoch = 8

            train(
                model=model,
                optimizer=optimizer,
                num_epoch=num_epoch,
                dict_loader=dict_loader,
                criterion=criterion,
                metric=metric,
                scheduler=scheduler,
                device=device,
            )
            if save_train is not None:
                torch.save(model.state_dict(), save_train)
    else:
        test_path: List[str] = glob("data/idao_dataset/*_test/*.png")

        resnet18 = models.resnet18(pretrained=True)
        backbone = nn.Sequential(nn.Sequential(*list(resnet18.children())[:-2]))
        model = TwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)

        model.load_state_dict(torch.load(trained_model_path))
        model.eval()

        test_dataset = IdaoInferenceDataset(list_path=test_path)

        sub_maker = SubmissionMaker(
            eval_model=model,
            submission_template="data/idao_dataset/track1_predictions_example.csv",
            test_dataset=test_dataset,
            device="cuda:0",
        )
        sub_maker.infer(save_path="issou.csv", rounding=True)


if __name__ == "__main__":
    save_train = f"models/test.pth"
    main(mode="train", trained_model_path=save_train)
