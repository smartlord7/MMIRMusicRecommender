import os

import numpy as np
from os.path import isfile
from metrics.similarity import self_dist
from const import DELIMITER_FEATURE, EXTENSION_CSV, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, \
    WRAPPER_METADATA_ADJECTIVES, DELIMITER_METADATA_ADJECTIVES, EXTENSION_MP3, DELIMITER_METADATA_PROPERTIES, \
    PATH_METADATA, OUT_PATH_CONTEXT_SIMILARITY, DELIMITER_METADATA_SIMILARITY


def gen_distances(dist_func: str, in_path: str = OUT_PATH_ALL_FEATURES, out_dir_path: str = OUT_PATH_DISTANCES, features_matrix=None) -> None:
    """
    Generates all the distances given a file with all features.
    """
    file_name = out_dir_path + dist_func + EXTENSION_CSV
    if isfile(file_name):
        return

    print("[DEBUG] Calculating %s distances for %s ..." % (dist_func, in_path))
    if features_matrix is None:
        features_matrix = np.genfromtxt(in_path, delimiter=DELIMITER_FEATURE)

    distances = self_dist(features_matrix, dist_func)
    np.savetxt(file_name, distances, fmt="%f", delimiter=DELIMITER_FEATURE)


def rank_similarity_analysis(query_file_path: str, distances_file_path: str, database_path: str, n=21):
    """
    Function used to calculate the ranking of the results.
    """
    database_files = os.listdir(database_path)
    database_files.sort()
    query_name = query_file_path.split("/")[-1]

    query_index = database_files.index(query_name)
    all_dist = np.genfromtxt(distances_file_path, delimiter=DELIMITER_FEATURE)
    query_dist = all_dist[query_index]
    sorted_dist_idx = np.argsort(query_dist)
    top_n_distances_idx = sorted_dist_idx[: n]
    top_n_results = np.take(database_files, top_n_distances_idx)
    top_n_results_dist = np.take(query_dist, top_n_distances_idx)

    return top_n_results, top_n_results_dist


def objective_analysis(in_path: str = PATH_METADATA, out_path: str = OUT_PATH_CONTEXT_SIMILARITY, query: str = None, n: int = 20):
    if not query and isfile(out_path):
        return None

    with open(in_path) as f:
        metadata = [row.split(DELIMITER_METADATA_PROPERTIES) for row in f.readlines()]
        size = len(metadata)
        metadata = metadata[1: size]
        points = list()

        if query:
            query = query.strip(EXTENSION_MP3)
            print("[DEBUG] Calculating top %d ranking based on metadata of %s..." % (n, query))
        else:
            similarity_matrix = list()
            print("[DEBUG] Calculating similarity matrix based on metadata...")

        for row in metadata:
            if not query or row[0].strip(WRAPPER_METADATA_ADJECTIVES) == query:
                print("[DEBUG] Calculating similarity row of %s: %s" % (row[0], row[1]))

                for other_row in metadata:
                    count = 0

                    if not query or other_row[0].strip(WRAPPER_METADATA_ADJECTIVES) != query:

                        # ARTIST
                        if other_row[1].strip(WRAPPER_METADATA_ADJECTIVES) == row[1].strip(WRAPPER_METADATA_ADJECTIVES):
                            count += 1

                        # GENRE
                        genre = other_row[11].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES)
                        genre_2 = row[11].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES)

                        genre = set(list(map(lambda x: x.lower(), genre)))
                        genre_2 = set(list(map(lambda x: x.lower(), genre_2)))

                        count += len(genre.intersection(genre_2))

                        # QUADRANT
                        if other_row[3].strip("\n'") == row[3].strip("\n'"):
                            count += 1

                        # EMOTION
                        emotion = set(other_row[9].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES))
                        emotion_2 = set(row[9].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES))

                        count += len(emotion.intersection(emotion_2))

                    else:
                        count = -1

                    points.append(count)

            if not query:
                similarity_matrix.append(points)
                points = list()

        if query:
            top_index = np.argsort(np.array(points))[len(points):len(points) - n - 1:-1]
            result = list()

            for i in top_index:
                result.append((metadata[i][0], metadata[i][2], metadata[i][1], points[i]))

            return result

        else:
            similarity_matrix = np.array(similarity_matrix)
            np.savetxt(out_path, similarity_matrix, fmt="%.f", delimiter=DELIMITER_METADATA_SIMILARITY)

            return np.array(similarity_matrix)


def calc_precision(results1: list, results2: list):
    
    set1 = set(results1)
    set2 = set(results2)
    intersection = set1.intersection(set2)

    return len(intersection) / max(len(results1), len(results2)) * 100
