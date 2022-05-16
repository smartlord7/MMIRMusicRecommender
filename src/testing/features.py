import numpy as np
from os.path import isfile
from const import DELIMITER_FEATURE
from mmir_pipeline.process import normalize_min_max


def process_default_features(in_path: str,
                             out_path: str) -> np.ndarray:
    """
    Function used to normalize already computed features.
    :param in_path: The path of the .csv file that contains the features.
    :param out_path: The path of the file to which the normalized features will be writen.
    :return: The normalized features values.
    """

    matrix = np.genfromtxt(in_path, delimiter=DELIMITER_FEATURE)
    values = matrix[1:, 1:matrix.shape[1] - 1]  # exclude first column (musics codes) and first row (features names)
    values = normalize_min_max(values)

    if isfile(in_path):
        return values

    np.savetxt(out_path, values, fmt='%f', delimiter=DELIMITER_FEATURE)

    return values
