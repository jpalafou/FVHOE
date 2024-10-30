import numpy as np


def l1err(x: np.ndarray, y: np.ndarray) -> float:
    """
    args:
        x (array_like)
        y (array_like)
    returns:
        out (float) : L1 error between x and y
    """
    return np.mean(np.abs(x - y))


def l2err(x: np.ndarray, y: np.ndarray) -> float:
    """
    args:
        x (array_like)
        y (array_like)
    returns:
        out (float) : L2 error between x and y
    """
    return np.sqrt(np.mean(np.square(x - y)))
