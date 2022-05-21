# region Dependencies


import warnings
import numpy as np
from scipy.stats import stats
from pipeline.process import *
import features.root.spectral as frs
import features.root.temporal as frt
import features.librosa_wrap.misc as lwm
from metrics.correlation import correlate
import features.librosa_wrap.spectral as lws
import features.librosa_wrap.temporal as lwt
from pipeline.sim_analysis import gen_distances, \
    rank_by_sim_analysis, \
    objective_analysis, \
    calc_precision


# endregion Dependencies


# region Const


FUNCTIONS_STATISTICS = [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.max, np.min]
FUNCTIONS_FEATURES = [lws.calc_mfcc, lws.calc_centroid, lws.calc_bandwidth, lws.calc_contrast,
                      lws.calc_flatness, lws.calc_roll_off, lwt.calc_fundamental_freq, lwt.calc_rms,
                      lwt.calc_zero_crossing_rate, lwm.calc_tempo]
FUNCTIONS_ROOT_FEATURES = [frs.calc_mfcc, frs.calc_centroid, frs.calc_bandwidth, lws.calc_contrast,
                           frs.calc_flatness, frs.calc_roll_off, frt.calc_fundamental_freq, frt.calc_rms,
                           frt.calc_zero_crossing_rate, lwm.calc_tempo]


# endregion Const


# region Private Functions


def _setup() -> list:
    """
    Function that setups the working environment.
    :return: The .mp3 files used as MMIR queries.
    """
    warnings.filterwarnings("ignore")
    queries = os.listdir(PATH_QUERIES)

    return queries


def _process() -> np.ndarray:
    """
    Function used to process the data using librosa and root implemented functions.
    :return: The features that were already computed, normalized.
    """
    default_features = process_default_features(IN_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_FEATURES)
    print("Processing database using librosa based features...")
    process_data(FUNCTIONS_STATISTICS, FUNCTIONS_FEATURES,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_FEATURES)
    print("Processing database using root implemented features...")
    process_data(FUNCTIONS_STATISTICS, FUNCTIONS_ROOT_FEATURES,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_ROOT_FEATURES)

    return default_features


def _generate_distances(default_features) -> None:
    """
    Function that calculates the distances between the feature arrays.
    :param default_features: The features that were already computed, normalized.
    :return: None
    """
    for dist in TYPES_DISTANCES:
        gen_distances(dist)
        gen_distances(dist, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, default_features)
        gen_distances(dist, OUT_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_DISTANCES)
        gen_distances(dist, OUT_PATH_ALL_ROOT_FEATURES, OUT_PATH_ALL_ROOT_DISTANCES)


def _correlate_distances() -> None:
    """
    Function that computes some correlation-related statistics
    between the distances calculated using librosa and the ones using
    root-implemented functions.
    :return: None
    """
    for dist in TYPES_DISTANCES:
        file_suffix = dist + EXTENSION_CSV
        file1 = OUT_PATH_ALL_ROOT_DISTANCES + file_suffix
        file2 = OUT_PATH_DISTANCES + file_suffix
        r_list = correlate(OUT_PATH_ALL_ROOT_DISTANCES + file_suffix, OUT_PATH_DISTANCES + file_suffix)
        print("Correlation between distances from %s and %s: " % (file1, file2))
        print("Mean: %s" % np.mean(r_list))
        print("Std.: %s" % np.std(r_list))
        print("Max: %s" % np.max(r_list))
        print("Min.: %s" % np.min(r_list))


def _show_ranking(query: str,
                 obj_ids: list,
                 dist_func: str,
                 dist_file_path: str,
                 database_path: str) -> None:
    """
    Function that presents the ranking of recommendations for a query, for a given distance function.
    :param query: The query to be analysed.
    :param obj_ids: The ids of the database files, returned by the previously made objective analysis.
    :param dist_func: The distance function to use.
    :param dist_file_path: The file of the path that contains the similarity matrix calculated used the given distance function.
    :param database_path: The path of the directory that contains all the database files.
    :return: None
    """
    dist_file_name = dist_file_path + dist_func + EXTENSION_CSV
    print("[DEBUG] Ranking results for query %s based on '%s' distanced features (source = %s)" % (query, dist_func, dist_file_name))
    results_features, dist = rank_by_sim_analysis(query, dist_file_name, database_path)
    results_features_ids = list()

    for i in range(len(results_features)):
        results_features_ids.append(results_features[i].strip(EXTENSION_MP3))
        print("%d - %s (%.4f)" % (i + 1, results_features[i], dist[i]))
    print("Precision: %.2f" % calc_precision(results_features[1:], obj_ids))


def _analyse_similarity(queries: list) -> None:
    """
    Function used to analise the similarity, based on metadata and feature arrays..
    :param queries: The files to be analysed,
    :return: None
    """
    for query in queries:
        results_obj = objective_analysis(query=query)
        results_obj_ids = list(map(lambda x: x[0], results_obj))
        print("[DEBUG] Ranking results for query %s based on metadata" % query)

        for i in range(len(results_obj)):
            curr = results_obj[i]
            print("%d - %s: %d" % (i + 1, curr[0], curr[3]))

        for dist in TYPES_DISTANCES:
            _show_ranking(query, results_obj_ids, dist, OUT_PATH_ALL_ROOT_DISTANCES, IN_DIR_PATH_ALL_DATABASE)
            _show_ranking(query, results_obj_ids, dist, OUT_PATH_DISTANCES, IN_DIR_PATH_ALL_DATABASE)
            _show_ranking(query, results_obj_ids, dist, OUT_PATH_DEFAULT_DISTANCES, IN_DIR_PATH_ALL_DATABASE)


def _main() -> None:
    """
    Main function.
    """
    queries = _setup()
    objective_analysis()
    default_features = _process()
    _generate_distances(default_features)
    _analyse_similarity(queries)
    _correlate_distances()


# endregion Private Functions


if __name__ == '__main__':
    _main()
