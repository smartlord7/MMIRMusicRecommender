import os
import numpy as np
from os.path import isfile
from metrics.similarity import self_dist
from const import DELIMITER_FEATURE, EXTENSION_CSV, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, \
    WRAPPER_METADATA_ADJECTIVES, DELIMITER_METADATA_ADJECTIVES, EXTENSION_MP3, DELIMITER_METADATA_PROPERTIES, \
    PATH_METADATA, OUT_PATH_CONTEXT_SIMILARITY, DELIMITER_METADATA_SIMILARITY, COL_METADATA_MUSIC_ID, \
    COL_METADATA_ARTIST, COL_METADATA_GENRE, COL_METADATA_QUADRANT, COL_METADATA_EMOTIONS


def gen_distances(dist_func: str,
                  in_path: str = OUT_PATH_ALL_FEATURES,
                  out_dir_path: str = OUT_PATH_DISTANCES,
                  features_matrix: np.ndarray = None) -> None:
    """
    Function that Generates and logs the similarity matrix for a given feature matrix.
    :param dist_func: The function used to compute the distance between feature arrays.
    :param in_path: The path of the file that contains the feature matrix.
    :param out_dir_path: The path of the file to which the distances matrix will be written into.
    :param features_matrix:
    :return: None
    """
    file_name = out_dir_path + dist_func + EXTENSION_CSV
    if isfile(file_name):
        return

    print("[DEBUG] Calculating %s distances for %s ..." % (dist_func, in_path))
    if features_matrix is None:
        features_matrix = np.genfromtxt(in_path, delimiter=DELIMITER_FEATURE)

    distances = self_dist(features_matrix, dist_func)
    np.savetxt(file_name, distances, fmt="%f", delimiter=DELIMITER_FEATURE)


def rank_by_sim_analysis(query_file_path: str,
                         distances_file_path: str,
                         database_path: str,
                         n:int = 21) -> tuple:
    """
    Function used to calculate the ranking of the results based on the similarity analysis of the query and the database.
    :param query_file_path: The path of the file that contains the query data.
    :param distances_file_path: The path of the file that contains the similarity matrix.
    :param database_path: The path of the directory that contains the database objects.
    :param n: Specifies the number of ordered results shown in the ranking.
    :return: The top n results and correspondent distances.
    """
    database_files = os.listdir(database_path)
    database_files.sort()
    query_name = query_file_path.split("/")[-1]

    query_index = database_files.index(query_name)
    sim_matrix = np.genfromtxt(distances_file_path, delimiter=DELIMITER_FEATURE)
    query_dist = sim_matrix[query_index]
    sorted_dist_idx = np.argsort(query_dist)
    top_n_distances_idx = sorted_dist_idx[: n]
    top_n_results = np.take(database_files, top_n_distances_idx)
    top_n_results_dist = np.take(query_dist, top_n_distances_idx)

    return top_n_results, top_n_results_dist


def objective_analysis(in_path: str = PATH_METADATA,
                       out_path: str = OUT_PATH_CONTEXT_SIMILARITY,
                       query: str = None,
                       n: int = 20) -> list or np.ndarray:
    """
    Function used to perform the ranking of the top n results based on the context metadata or to calculate/log
    the similarity matrix based on the same data.
    :param in_path: The path of the file that contains all the metadata.
    :param out_path: The path of the file to which the context-based similarity matrix will be written into.
    :param query: The name of the query (MTXXXXX)
    :param n: Specifies the number of ordered results shown in the ranking.
    :return: A list containing the top n ranking results or the context-based similarity matrix.
    """
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
            if not query or row[COL_METADATA_MUSIC_ID].strip(WRAPPER_METADATA_ADJECTIVES) == query:
                print("[DEBUG] Calculating similarity row of %s: %s" % (row[0], row[1]))

                for other_row in metadata:
                    count = int()

                    if not query or other_row[COL_METADATA_MUSIC_ID].strip(WRAPPER_METADATA_ADJECTIVES) != query:

                        # ARTIST
                        count += int(other_row[COL_METADATA_ARTIST].strip(WRAPPER_METADATA_ADJECTIVES) == row[COL_METADATA_ARTIST].strip(WRAPPER_METADATA_ADJECTIVES))

                        # QUADRANT
                        count += int(other_row[COL_METADATA_QUADRANT].strip("\n'") == row[COL_METADATA_QUADRANT].strip("\n'"))

                        # EMOTION
                        query_emotion = set(row[COL_METADATA_EMOTIONS].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES))
                        emotion = set(other_row[COL_METADATA_EMOTIONS].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES))
                        count += len(emotion.intersection(query_emotion))

                        # GENRE
                        query_genre = row[COL_METADATA_GENRE].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES)
                        genre = other_row[COL_METADATA_GENRE].strip(WRAPPER_METADATA_ADJECTIVES).split(DELIMITER_METADATA_ADJECTIVES)

                        query_genre = set(list(map(lambda x: x.lower(), query_genre)))
                        genre = set(list(map(lambda x: x.lower(), genre)))

                        count += len(genre.intersection(query_genre))
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

            return similarity_matrix


def calc_precision(results1: list,
                   results2: list) -> float:
    """
    Function that calculates the precision between two sets of results: the retrieved and the relevant ones.
    :param results1: The retrieved results.
    :param results2: The relevant results.
    :return:
    """
    set1 = set(results1)
    set2 = set(results2)
    intersection = set1.intersection(set2)

    return len(intersection) / max(len(results1), len(results2)) * 100
