import os

import numpy as np
from os.path import isfile
from metrics.similarity import self_dist
from const import FEATURE_DELIM, EXTENSION_CSV, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES


def gen_distances(dist_func: str, in_path: str = OUT_PATH_ALL_FEATURES, out_dir_path: str = OUT_PATH_DISTANCES, features_matrix=None) -> None:
    file_name = out_dir_path + dist_func + EXTENSION_CSV
    if isfile(file_name):
        return

    print("Calculating %s distances for %s ..." % (dist_func, in_path))
    if features_matrix is None:

        features_matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)

    distances = self_dist(features_matrix, dist_func)
    np.savetxt(file_name, distances, fmt="%f", delimiter=FEATURE_DELIM)


def rank(query_file_path: str, distances_file_path: str, database_path: str, n=20):
    database_files = os.listdir(database_path).sort()

