import numpy as np


def mse(x, y):
    return np.mean(np.square(x - y))
