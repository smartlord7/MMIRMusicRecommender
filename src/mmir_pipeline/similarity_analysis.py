from os.path import isfile

import numpy as np
from metrics.similarity import self_dist
from const import FEATURE_DELIM, EXTENSION_CSV


def gen_distances(in_path: str, out_dir_path: str, dist_func: str) -> None:
    file_name = out_dir_path + dist_func + EXTENSION_CSV
    if isfile(file_name):
        return

    print("Calculating %s distances..." % dist_func)

    features_matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)
    distances = self_dist(features_matrix, dist_func)
    np.savetxt(file_name, distances, fmt="%f", delimiter=FEATURE_DELIM)
