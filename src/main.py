import numpy as np


FEATURE_DELIM = ","
TOP100_FEATURES_PATH = "data/features/top100_features.csv"


def min_max_normalize(matrix, a=0, b=1):
    min_val = matrix.min() * np.ones(matrix.shape)
    max_val = matrix.max() * np.ones(matrix.shape)
    a = a * np.ones(matrix.shape)
    b = b * np.ones(matrix.shape)

    matrix = (a + (matrix - min_val) * (b - a)) / (max_val - min_val)

    return matrix


def process_features(path):
    matrix = np.genfromtxt(path, delimiter=FEATURE_DELIM)
    matrix = matrix[1:, 1:matrix.shape[1] - 1]

    matrix = min_max_normalize(matrix)

    return matrix


def main():
    print(process_features(TOP100_FEATURES_PATH))


if __name__ == '__main__':
    main()