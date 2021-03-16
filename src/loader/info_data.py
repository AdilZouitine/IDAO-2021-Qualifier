import re
from typing import Dict

DICT_CLASS: Dict[str, str] = {"ER": 1, "NR": 0}


def get_particule_class(path: str) -> str:
    # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
    # ER
    class_name = re.search(r"(ER|NR)", path).group(0)
    return DICT_CLASS[class_name]


def get_particule_energy(path: str) -> int:
    # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_keV_930V_30cm_IDAO_iso_crop_hist_pic_run5_ev234;1.png
    # data/track_1/idao_dataset/train/ER/-0.0128632215783__CYGNO_60_40_ER_3_
    particule_kev = re.findall(r"\d*_keV", path)[0]
    particule_energy = re.findall(r"\d*", particule_kev)[0]
    return int(particule_energy)


def get_image_name(path: str) -> str:
    path = re.sub(r"\\", "/", path)  # for windows user
    image_name_and_extension = path.split("/")[-1]
    image_name = image_name_and_extension.split(".")[0]
    return image_name


def get_image_leaderboard(path: str) -> str:
    image_leaderboard = re.findall(r"(private|public)", path)[0]
    return image_leaderboard