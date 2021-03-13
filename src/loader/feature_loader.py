from PIL import Image
import numpy as np
from torch.utils.data import Dataset


from typing import Callable, Dict, List, NoReturn, Optional, Tuple
from .feature_calculator import compute_feature

from .info_data import (
    DICT_CLASS,
    get_image_name,
    get_particule_class,
    get_particule_energy,
    get_image_leaderboard,
)


class IdaoFeaturesDataset(Dataset):
    def __init__(self, list_path: List[str]) -> NoReturn:
        self.list_path = list_path

    def __len__(self) -> int:
        return len(self.list_path)

    def __getitem__(self, index: int) -> Tuple[dict, int, float]:
        image_path = self.list_path[index]
        image = np.array(Image.open(image_path), dtype=np.float32)

        particule_class = get_particule_class(path=image_path)
        particule_energy = get_particule_energy(path=image_path)
        image_feature = compute_feature(image)
        return (image_feature, particule_class, particule_energy)


class IdaoFeaturesInferenceDataset(IdaoFeaturesDataset):
    def __getitem__(self, index: int) -> Tuple[np.ndarray, str, str]:
        image_path = self.list_path[index]
        image_path = self.list_path[index]
        image = np.array(Image.open(image_path), dtype=np.float32)
        image_name = get_image_name(path=image_path)
        image_leaderboard = get_image_leaderboard(path=image_path)
        image_feature = compute_feature(image)
        return image_feature, image_name, image_leaderboard