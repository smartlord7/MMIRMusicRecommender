import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
from sklearn.metrics import pairwise_distances


def euclidian_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Function that calculates the euclidian distance between two arrays.
    :param array1: The first array.
    :param array2: The second array.
    :return: The euclidian distance between both arrays.
    """
    return ((array1 - array2) ** 2).sum()


def manhattan_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Function that calculates the Manhattan distance between two arrays.
    :param array1: The first array.
    :param array2: The second array.
    :return: The Manhattan distance between both arrays.
    """
    return cityblock(array1, array2)


def cosine_dist(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Function that calculates the cosine distance between two arrays.
    :param array1: The first array.
    :param array2: The second array.
    :return: The cosine distance between both arrays.
    """
    return cosine(array1, array2)


def self_dist(matrix: np.ndarray, type_metric: str) -> np.ndarray:
    """
    Function that calculates the correspondent similarity matrix of a given matrix.
    :param matrix: The matrix from which the distances between rows will be calculated.
    :param type_metric: The type of metric to use when calculating the pairwise distances.
    :return:
    """
    return pairwise_distances(matrix, metric=type_metric)
