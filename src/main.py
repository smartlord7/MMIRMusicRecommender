import warnings

import numpy as np
from scipy.stats import stats
from pipeline.process import *
import features.librosa_wrap.misc as lwm
import features.librosa_wrap.spectral as lws
import features.librosa_wrap.temporal as lwt
import features.root.spectral as frs
import features.root.temporal as frt
from metrics.correlation import correlate
from pipeline.sim_analysis import gen_distances, \
    rank_by_sim_analysis, \
    objective_analysis, \
    calc_precision

FUNCTIONS_STATISTICS = [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.max, np.min]
FUNCTIONS_FEATURES = [lws.calc_mfcc, lws.calc_centroid, lws.calc_bandwidth, lws.calc_contrast,
                      lws.calc_flatness, lws.calc_roll_off, lwt.calc_fundamental_freq, lwt.calc_rms,
                      lwt.calc_zero_crossing_rate, lwm.calc_tempo]
FUNCTIONS_ROOT_FEATURES = [frs.calc_mfcc, frs.calc_centroid, frs.calc_bandwidth, lws.calc_contrast,
                           frs.calc_flatness, frs.calc_roll_off, frt.calc_fundamental_freq, frt.calc_rms,
                           frt.calc_zero_crossing_rate, lwm.calc_tempo]


def setup():
    """
    Setup function.
    :return: The queries.
    """
    warnings.filterwarnings("ignore")
    queries = os.listdir(PATH_QUERIES)

    return queries


def process():
    """
    Function used to process data.
    :return: the default features.
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


def generate_distances(default_features):
    """
    Function used to generate the distances.
    :param default_features: the default features.
    :return:
    """
    for dist in TYPES_DISTANCES:
        gen_distances(dist)
        gen_distances(dist, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, default_features)
        gen_distances(dist, OUT_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_DISTANCES)
        gen_distances(dist, OUT_PATH_ALL_ROOT_FEATURES, OUT_PATH_ALL_ROOT_DISTANCES)


def correlate_distances():
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


def show_ranking(query: str, obj_ids: list, dist_func: str, dist_file_path: str, database_path: str):
    dist_file_name = dist_file_path + dist_func + EXTENSION_CSV
    print("[DEBUG] Ranking results for query %s based on '%s' distanced features (source = %s)" % (query, dist_func, dist_file_name))
    results_features, dist = rank_by_sim_analysis(query, dist_file_name, database_path)
    results_features_ids = list()

    for i in range(len(results_features)):
        results_features_ids.append(results_features[i].strip(EXTENSION_MP3))
        print("%d - %s (%.4f)" % (i + 1, results_features[i], dist[i]))
    print("Precision: %.2f" % calc_precision(results_features[1:], obj_ids))


def analyse_similarity(queries):
    """
    Function used to analise the similarities.
    :param queries: are the queries to be compared.
    :return:
    """
    for query in queries:
        results_obj = objective_analysis(query=query)
        results_obj_ids = list(map(lambda x: x[0], results_obj))
        print("[DEBUG] Ranking results for query %s based on metadata" % query)

        for i in range(len(results_obj)):
            curr = results_obj[i]
            print("%d - %s: %d" % (i + 1, curr[0], curr[3]))

        for dist in TYPES_DISTANCES:
            show_ranking(query, results_obj_ids, dist, OUT_PATH_ALL_ROOT_DISTANCES, IN_DIR_PATH_ALL_DATABASE)
            show_ranking(query, results_obj_ids, dist, OUT_PATH_DISTANCES, IN_DIR_PATH_ALL_DATABASE)
            show_ranking(query, results_obj_ids, dist, OUT_PATH_DEFAULT_DISTANCES, IN_DIR_PATH_ALL_DATABASE)



def main():
    """
        Main function.
    """
    queries = setup()
    objective_analysis()
    default_features = process()
    generate_distances(default_features)
    analyse_similarity(queries)
    correlate_distances()


if __name__ == '__main__':
    main()
