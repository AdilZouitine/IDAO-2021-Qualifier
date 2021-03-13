from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .info_data import (
    DICT_CLASS,
    get_image_name,
    get_particule_class,
    get_particule_energy,
    get_image_leaderboard,
)


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
        particule_class = get_particule_class(path=image_path)
        # particule_angle = self.get_particule_angle(path=image_path)
        particule_energy = get_particule_energy(path=image_path)
        return (
            tensor_image,
            torch.tensor(particule_class, dtype=torch.float32).unsqueeze(dim=-1),
            # torch.tensor(particule_angle).unsqueeze(dim=-1),
            torch.tensor(particule_energy).unsqueeze(dim=-1),
        )


class IdaoInferenceDataset(IdaoDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        image_path = self.list_path[index]
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image)
        image_name = get_image_name(path=image_path)
        image_leaderboard = get_image_leaderboard(path=image_path)
        return tensor_image, image_name, image_leaderboard
