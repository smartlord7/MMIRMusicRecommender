import numpy as np
from const import *


def min_max_normalize(matrix, a=0, b=1):
    min_val = matrix.min()
    max_val = matrix.max()

    matrix = (a + (matrix - min_val) * (b - a)) / (max_val - min_val)

    return matrix


def process_features(in_path, out_path):
    matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)
    values = matrix[1:, 1:matrix.shape[1] - 1]

    values = min_max_normalize(values)
    np.savetxt(out_path, values, delimiter=FEATURE_DELIM)

    return values


def main():
    process_features(IN_PATH_ORIGINAL_FEATURES, OUT_PATH_ORIGINAL_FEATURES)


if __name__ == '__main__':
    main()
