import numpy as np
from const import *
from util.process import *


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
