import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock


def euclidian_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return ((array1 - array2) ** 2).sum()


def manhattan_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return cityblock(array1, array2)


def cosine_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return cosine(array1, array2)
