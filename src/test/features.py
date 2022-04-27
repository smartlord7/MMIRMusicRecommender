import numpy as np
from const import FEATURE_DELIM
from util.process import normalize_min_max


def process_default_features(in_path, out_path):
    """
    Function used to process the used features.
    :param in_path: the input directory.
    :param out_path: the output directory.
    :return: required values.
    """
    matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)
    values = matrix[1:, 1:matrix.shape[1] - 1]

    values = normalize_min_max(values)
    np.savetxt(out_path, values, delimiter=FEATURE_DELIM)

    return values