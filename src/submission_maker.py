from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from loader.image_loader import DICT_CLASS, IdaoInferenceDataset
from loss import round_kev, update_class


class SubmissionMaker:
    def __init__(
        self,
        eval_model: nn.Module,
        submission_template: str,
        test_dataset: IdaoInferenceDataset,
        device: str = "cuda:0",
    ):
        """[Use trained model to make prediction and generate a submission file]

        Args:
            eval_model (nn.Module): [Eval model after loading weight to architecture]
            submission_template (str): [Submission template file]
            test_dataset (IdaoInferenceDataset): [Pytorch dataset for private and public test]
            device (str): [CPU/GPU device]
        """

        self.eval_model = eval_model.to(device)
        self.submission = pd.read_csv(submission_template)
        self.test_dataset = test_dataset
        self.device = device

    def infer(
        self,
        save_path: str,
        rounding: Optional[bool] = True,
        update_class: Optional[bool] = False,
    ) -> NoReturn:
        """[Infer a model and create a submission file]

        Args:
            save_path (str): [Location to save the submsission csv file]
            rounding (Optional[bool], optional): [Round the energy]. Defaults to True.

        Returns:
            NoReturn: [description]
        """
        for image, image_name, image_leaderboard in tqdm(self.test_dataset):
            image = image.to(self.device).unsqueeze(0)
            predicted_class, predicted_kev = self.eval_model(image)
            # reformat
            predicted_class = predicted_class.squeeze(0).detach().cpu().numpy()
            predicted_kev = predicted_kev.squeeze(0).detach().cpu().numpy()

            if rounding:
                if image_leaderboard == "public":
                    targets = {
                        DICT_CLASS["ER"]: [3, 10, 30],
                        DICT_CLASS["NR"]: [1, 6, 20],
                    }

                else:
                    targets = {
                        DICT_CLASS["ER"]: [1, 6, 20],
                        DICT_CLASS["NR"]: [3, 10, 30],
                    }

                predicted_class = update_class(
                    predicted_class=predicted_class, predicted_kev=predicted_kev,
                )

                predicted_kev = round_kev(
                    predicted_kev=predicted_kev,
                    predicted_class=predicted_class,
                    targets=targets,
                )

            predicted_class = np.rint(predicted_class)

            self.submission.loc[
                self.submission["id"] == image_name, "classification_predictions"
            ] = int(predicted_class)

            self.submission.loc[
                self.submission["id"] == image_name, "regression_predictions"
            ] = int(predicted_kev)

        self.submission.to_csv(path_or_buf=save_path, sep=",", index=False)
