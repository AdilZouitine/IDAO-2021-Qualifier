from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import re

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
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image)
        particule_class = self.get_particule_class(path=image_path)
        # particule_angle = self.get_particule_angle(path=image_path)
        particule_energy = self.get_particule_energy(path=image_path)
        return (
            tensor_image,
            torch.tensor(particule_class, dtype=torch.float32).unsqueeze(dim=-1),
            # torch.tensor(particule_angle).unsqueeze(dim=-1),
            torch.tensor(particule_energy).unsqueeze(dim=-1),
        )

    @classmethod
    def get_particule_class(cls, path: str) -> str:
        # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
        # ER
        class_name = re.search(r"(ER|NR)", path).group(0)
        return DICT_CLASS[class_name]

    # @classmethod
    # def get_particule_angle(cls, path: str) -> float:
    #     # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
    #     # -0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
    #     name_file = path.split("/")[-1]
    #     # -0.0128632215783
    #     particule_angle = float(name_file.split("__")[0])
    #     return particule_angle

    @classmethod
    def get_particule_energy(cls, path: str) -> int:
        # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
        # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_
        particule_kev = re.findall(r"\d*_keV", path)[0]
        particule_energy = re.findall(r"\d*", particule_kev)[0]
        return int(particule_energy)


class IdaoInferenceDataset(IdaoDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_path = self.list_path[index]
        image = Image.open(image_path).convert("RGB")
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
