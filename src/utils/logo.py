from typing import List, Tuple

import pandas as pd


def logo(list_files: List[str]) -> Tuple[List[str], List[str]]:
    """Leave one group out. Yield a tuple of lists. 
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

    groups = ["10ER", "1NR", "20NR", "30ER", "3ER", "6NR"]
    groups_val = ["10NR", "1ER", "20ER", "30NR", "3NR", "6ER"]

    for validation_group in groups:

        fit = [group for group in groups if group != validation_group]
        val = [validation_group] + groups_val
        yield data[data["group"].isin(fit)]["path"].to_list(), data[
            data["group"].isin(val)
        ]["path"].to_list()

