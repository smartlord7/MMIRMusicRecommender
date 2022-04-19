import numpy as np
from const import *
from features.spectral_features import *
from features.temporal_features import *


def min_max_normalize(matrix, a=0, b=1):
    """
    Given one matrix,this function will normalize it within the min and max values..
    :param matrix: The used matrix.
    :param a: Min value
    :param b: Max value
    :return: normalized matrix.
    """
    min_val = matrix.min()
    max_val = matrix.max()

    matrix = (a + (matrix - min_val) * (b - a)) / (max_val - min_val)

    return matrix


def process_features(in_path, out_path):
    """
    Function used to process the used features.
    :param in_path: the input directory.
    :param out_path: the output directory.
    :return: required values.
    """
    matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)
    values = matrix[1:, 1:matrix.shape[1] - 1]

    values = min_max_normalize(values)
    np.savetxt(out_path, values, delimiter=FEATURE_DELIM)

    # cent = centroid(values)
    # flat = flatness(values)
    # rms = rms(values)

    return values


def main():
    """
    Main function.
    """
    process_features(IN_PATH_ORIGINAL_FEATURES, OUT_PATH_ORIGINAL_FEATURES)


if __name__ == '__main__':
    main()
