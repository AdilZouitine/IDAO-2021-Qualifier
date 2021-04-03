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
from utils.logo import logo
from utils.dummy_validation import dummy_validation
from submission_maker import SubmissionMaker


train_path: List[str] = glob("../data/track_1/idao_dataset/new_test/train/*/*.png")
test_path: List[str] = glob("../data/track_1/idao_dataset/new_test/*_test/*.png")

transform = transforms.Compose(
    [
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(
            (-0.3918, -0.2711, -0.0477),
            (0.0822, 0.0840, 0.0836),  # mean and std over all the dataset
        ),
    ]
)

fold_generator = logo(list_files=train_path)
# fold_generator = dummy_validation(list_files=train_path)
device = torch.device("cuda")

for index, (fold_train_path, fold_val_path) in enumerate(fold_generator):

    train_dataset = IdaoDataset(list_path=fold_train_path, transform=transform)
    val_dataset = IdaoDataset(list_path=fold_val_path, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=50, shuffle=True, num_workers=8
    )

    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=50, shuffle=False, num_workers=8
    )

    resnet18 = models.resnet18(pretrained=True)
    backbone = nn.Sequential(nn.Sequential(*list(resnet18.children())[:-2]))
    # model = TwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)
    model = IndependantTwoHeadModel(backbone_model=backbone, embedding_size=512 * 7 * 7)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=5e-5)
    criterion = IdaoLoss(w_classification=1, w_regression=1)
    metric = partial(idao_metric, do_round_kev=True, do_update_class=True)  # Callable
    scheduler = None

    dict_loader = {"train": train_dataloader, "validation": val_dataloader}
    num_epoch = 2

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
    model.eval()

    test_dataset = IdaoInferenceDataset(list_path=test_path, transform=transform)

    sub_maker = SubmissionMaker(
        eval_model=model,
        submission_template="../data/track1_predictions_example_2.csv",
        test_dataset=test_dataset,
        device="cuda",
    )
    print("Inference")
    sub_maker.infer(
        save_path=f"../result/dummy_lg_pred_da_new_2{index}.csv", rounding=True
    )
    torch.save(model.state_dict(), f"../result/model_{index}.pth")
