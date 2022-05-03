import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
from sklearn.metrics import pairwise_distances


def euclidian_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return ((array1 - array2) ** 2).sum()


def manhattan_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return cityblock(array1, array2)


def cosine_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    return cosine(array1, array2)


def self_dist(matrix: np.ndarray, type_metric: str) -> np.ndarray:
    return pairwise_distances(matrix, metric=type_metric)

