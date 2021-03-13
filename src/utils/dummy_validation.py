from typing import List, Tuple
from sklearn.model_selection import KFold
import pandas as pd


def dummy_validation(
    list_files: List[str], random_state: int = 7, n_splits: int = 5
) -> Tuple[List[str], List[str]]:
    """[Dummy validation, we valid our model on the same class of particle]

    Args:
        list_files (List[str]): [List of all train file]
        random_state (int, optional): [description]. Defaults to 7.
        n_splits (int, optional): [description]. Defaults to 5.

    Returns:
        Tuple[List[str], List[str]]: [description]

    Yields:
        Iterator[Tuple[List[str], List[str]]]: [description]
    """

    data = []
    kf = KFold(n_splits=n_splits, shuffle=True)

    for file in list_files:
        metadata = file.split("keV_")[0].split("_")
        group = f"{metadata[-2]}{metadata[-3]}"
        data.append([group, file])

    data = pd.DataFrame(data, columns=["group", "path"])

    groups = ["10ER", "1NR", "20NR", "30ER", "3ER", "6NR"]

    sub_data = data[data["group"].isin(groups)]["path"].to_numpy()
    for train_idx, val_idx in kf.split(sub_data):
        yield sub_data[train_idx], sub_data[val_idx]
