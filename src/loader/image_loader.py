from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

DICT_CLASS: Dict[str, str] = {"ER": 0, "NR": 1}


class IdaoDataset(Dataset):
    def __init__(
        self, list_path: List[str], transform: Optional[Callable] = None
    ) -> NoReturn:
        self.list_path = list_path
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.list_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, float]:
        image_path = self.list_path[index]
        image = Image.open(image_path)
        tensor_image = self.transform(image)
        particule_class = self.get_particule_class(path=image_path)
        particule_angle = self.get_particule_angle(path=image_path)
        return tensor_image, particule_class, particule_angle

    @classmethod
    def get_particule_class(cls, path: str) -> str:
        # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
        # ER
        class_name = path.split("/")[-2]
        return DICT_CLASS[class_name]

    @classmethod
    def get_particule_angle(cls, path: str) -> float:
        # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
        # -0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
        name_file = path.split("/")[-1]
        # -0.0128632215783
        particule_angle = float(name_file.split("__")[0])
        return particule_angle


class IdaoInferenceDataset(IdaoDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_path = self.list_path[index]
        image = Image.open(image_path)
        tensor_image = self.transform(image)
        image_name = self.get_image_name(path=image_path)
        return tensor_image, image_name

    @classmethod
    def get_image_name(cls, path: str) -> str:
        # /track_1/idao_dataset/private_test/0a0b46086a0a870eacfdd03622bfc3c79705773f.png
        image_name_and_extension = path.split("/")[-1]
        # 0a0b46086a0a870eacfdd03622bfc3c79705773f.png
        image_name = image_name_and_extension.split(".")[0]
        # 0a0b46086a0a870eacfdd03622bfc3c79705773f
        return image_name
