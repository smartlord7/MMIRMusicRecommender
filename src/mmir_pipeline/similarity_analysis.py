import os

import numpy as np
from os.path import isfile
from metrics.similarity import self_dist
from const import FEATURE_DELIM, EXTENSION_CSV, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES


def gen_distances(dist_func: str, in_path: str = OUT_PATH_ALL_FEATURES, out_dir_path: str = OUT_PATH_DISTANCES, features_matrix=None) -> None:
    """
    Generates all the distances given a file with all features.
    """
    file_name = out_dir_path + dist_func + EXTENSION_CSV
    if isfile(file_name):
        return

    print("Calculating %s distances for %s ..." % (dist_func, in_path))
    if features_matrix is None:

        features_matrix = np.genfromtxt(in_path, delimiter=FEATURE_DELIM)

    distances = self_dist(features_matrix, dist_func)
    np.savetxt(file_name, distances, fmt="%f", delimiter=FEATURE_DELIM)


def rank_query_results(query_file_path: str, distances_file_path: str, database_path: str, n=20):
    """
    Function used to calculate the ranking of the results.
    """
    database_files = os.listdir(database_path)
    database_files.sort()
    query_name = query_file_path.split("/")[-1]

    print("Ranking results for query %s based on distances in %s" % (query_name, distances_file_path))
    query_index = database_files.index(query_name)
    all_dist = np.genfromtxt(distances_file_path, delimiter=FEATURE_DELIM)
    query_dist = all_dist[query_index]
    sorted_dist_idx = np.argsort(query_dist)
    top_n_distances_idx = sorted_dist_idx[: n]
    top_n_results = np.take(database_files, top_n_distances_idx)
    top_n_results_dist = np.take(query_dist, top_n_distances_idx)

    return top_n_results, top_n_results_dist


