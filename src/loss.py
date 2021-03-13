import collections
from typing import Dict, List, NoReturn, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, roc_auc_score


class IdaoLoss(nn.Module):
    """[summary].

    Args:
        nn ([type]): [description]
    """

    def __init__(self, w_classification: float = 1, w_regression: float = 1):
        """[summary].

        Args:
            w_classification (float, optional): [description]. Defaults to 1.
            w_regression (float, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.regression_loss = nn.L1Loss()
        self.classification_loss = nn.BCELoss()
        self.w_classification = w_classification
        self.w_regression = w_regression

    def __call__(self, predicted_class, true_class, predicted_angle, true_angle):
        classification_loss = self.classification_loss(
            predicted_class, true_class
        )  # noqa: E501
        regression_loss = self.regression_loss(predicted_angle, true_angle)
        loss = (
            self.w_classification * classification_loss
            + self.w_regression * regression_loss
        )
        return loss


def idao_metric(
    predicted_class: List[torch.Tensor],
    true_class: List[torch.Tensor],
    predicted_kev: List[torch.Tensor],
    true_kev: List[torch.Tensor],
    do_update_class=False,
    do_round_kev=False,
):
    """[summary].

    Args:
        predicted_class (torch.Tensor): [description]
        true_class (torch.Tensor): [description]
        predicted_kev (torch.Tensor): [description]
        true_kev (torch.Tensor): [description]

    Returns:
        float: [description]
    """

    def __merge(list_tensor: List[torch.Tensor]):
        list_tensor = [tensor.view(-1) for tensor in list_tensor]
        return torch.cat(list_tensor)

    predicted_class = __merge(predicted_class)
    true_class = __merge(true_class)
    predicted_kev = __merge(predicted_kev)
    true_kev = __merge(true_kev)

    predicted_class = predicted_class.numpy()
    true_class = true_class.numpy()
    predicted_kev = predicted_kev.numpy()
    true_kev = true_kev.numpy()

    if do_update_class:
        predicted_class = update_class(
            predicted_class=predicted_class,
            predicted_kev=predicted_kev,
            true_class=true_class,
            true_kev=true_kev,
        )

    if do_round_kev:
        predicted_kev = round_kev(
            predicted_kev=predicted_kev,
            predicted_class=predicted_class,
            true_class=true_class,
            true_kev=true_kev,
        )

    classif = roc_auc_score(true_class, predicted_class,)
    regression = mean_absolute_error(true_kev, predicted_kev)
    return float((classif - regression) * 1000), classif, regression


def update_class(
    predicted_class: np.ndarray,
    predicted_kev: np.ndarray,
    true_class=None,
    true_kev=None,
):
    """Eval if there are too many predicted class 0 or class 1.
    If there are too many predicted class 0, replace nearest probability to 0.5 (< 0.5) by 0.51.
    If there are too many predicted class 1, replace nearest probability to 0.5 (> 0.5) by 0.49.
    
    >>> import numpy as np
    
    Too many class 0 predicted:
    >>> true_kev = np.array([1, 4, 5, 1, 3, 3])
    >>> true_class = np.array([1, 1, 1, 1, 0, 0]) 

    >>> predicted_kev = np.array([1, 3, 2, 1, 3, 2])
    >>> predicted_class = np.array([0.9, 0.49, 0.49, 0, 0, 0])
    
    >>> update_class(
    ...     predicted_class, 
    ...     predicted_kev, 
    ...     true_class, 
    ...     true_kev
    ... )
    array([0.9, 0.51, 0.51, 0.  , 0.51, 0.  ])
    
    Too many class 1 predicted:
    >>> true_kev = np.array([1, 4, 5, 1, 3, 3])
    >>> true_class = np.array([0, 0, 0, 0, 1, 1]) 

    >>> predicted_kev = np.array([1, 3, 2, 1, 3, 2])
    >>> predicted_class = np.array([0.1, 0.51, 0.51, 1., 1., 1.])
    
    >>> update_class(
    ...     predicted_class, 
    ...     predicted_kev, 
    ...     true_class, 
    ...     true_kev
    ... )
    array([0.49, 0.49, 0.49, 0.49, 1.  , 1.  ])

    >>> update_class(
    ...     predicted_class, 
    ...     predicted_kev, 
    ... )
    
    """
    if true_class is None and true_kev is None:
        targets = {
            1: {10: 2511, 30: 2511, 3: 2511, 1: 251, 20: 251, 6: 251},  # NR
            0: {1: 2511, 20: 2511, 6: 2511, 10: 251, 30: 251, 3: 251},  # ER
        }
    else:
        targets = {1: collections.defaultdict(int), 0: collections.defaultdict(int)}
        for c, kev in zip(true_class, true_kev):
            targets[c][kev] += 1

    n_class_0, n_class_1 = sum(targets[0].values()), sum(targets[1].values())
    n_predicted_1 = sum(np.rint(predicted_class))
    n_predicted_0 = len(predicted_class) - n_predicted_1

    # Too much class_1
    if n_predicted_1 > n_class_1:
        k = int(n_predicted_1 - n_class_1)
        k_predicted_class = abs(
            (predicted_class + 1e-5) * (predicted_class < 0.5) + 0.5
        )
        for i in np.argsort(k_predicted_class)[:k]:
            predicted_class[i] = 0.49

    # Too much class_0
    elif n_predicted_1 < n_class_1:
        k = int(n_predicted_0 - n_class_0)
        k_predicted_class = abs(
            (predicted_class + 1e-5) * (predicted_class < 0.5) - 0.5
        )
        for i in np.argsort(k_predicted_class)[:k]:
            predicted_class[i] = 0.51

    return predicted_class


def round_kev(
    predicted_kev: np.ndarray,
    predicted_class: np.ndarray,
    targets: Optional[Dict[int, List[int]]] = None,
    true_class=None,
    true_kev=None,
):
    """
    Find the nearest neighbor between the predicted class and the potential KeV.
    
    
    Parameters:
    -----------
        pedict_kev: Predicted KeV.
        prediced_class: Predicted probabilities to belong to classe 1.
        true_class: Target class. Only while validating, leaving it empy for testing.
        true_kev: Target KeV. Only while validating, leaving it empy for testing.
        targets: Mapping between classes and potential KeV.
        
    Example:
    --------
    
    While validating:
    
    >>> round_kev(
    ...    predicted_kev = np.array([7, 20, 20]),
    ...    predicted_class = np.array([0.1, 0.8, 0.7]),
    ...    true_class = np.array([0, 1, 1]),
    ...    true_kev = np.array([1, 10, 10]),
    ... )
    array([30, 10, 10])
    
    While predicting test, supposing: 
    ER = HE = class 0
    NR = E = class 1
    
    >>> round_kev(
    ...    predicted_kev = np.array([7, 20, 20]),
    ...    predicted_class = np.array([0.1, 0.8, 0.7]),
    ...    targets = {0: [3, 10, 30], 1: [1, 6, 20]}
    ... )
    array([30, 10, 10])
    
    """
    if targets is None:

        targets = collections.defaultdict(list)

        if true_class is not None and true_kev is not None:
            for c, k in zip(true_class, true_kev):
                c = c.item()
                k = k.item()
                if k not in targets[c]:
                    targets[c].append(k)
        else:
            raise ValueError(
                "You must pass true_class and true_kev, when validating your model and targets while predicting on test."
            )

    rounded_kev = []

    for p_kev, p_c in zip(predicted_kev, np.rint(predicted_class)):
        rounded_kev.append(min(targets[p_c], key=lambda y: abs(y - p_kev)))
    return np.array(rounded_kev)

