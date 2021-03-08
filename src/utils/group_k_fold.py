from typing import List, Tuple

import pandas as pd
from sklearn import model_selection


def group_k_fold(list_files: List[str], n_splits: int) -> Tuple[List[str], List[str]]:
    """Yield a tuple of lists. 
    The first list contains paths of the training set. 
    The second list contains paths of the validation set.
    
    Parameters:
    -----------
        list_files: Path of the training set.
        
    """
    data = []

    for file in list_files:
        metadata = file.split("keV_")[0].split("_")
        group = f"{metadata[-2]}{metadata[-3]}"
        data.append([group, file])

    data = pd.DataFrame(data, columns=["group", "path"])

    kfold = model_selection.GroupKFold(n_splits=n_splits)
    kfold.get_n_splits(data, groups=data["group"])
    for fit, val in kfold.split(data, groups=data["group"].copy()):
        yield data.loc[fit]["path"].to_list(), data.loc[val]["path"].to_list()

