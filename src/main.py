import warnings
from scipy.stats import stats
from pipeline.process import *
import features.librosa_wrap.misc as lwm
import features.librosa_wrap.spectral as lws
import features.librosa_wrap.temporal as lwt
import features.root.spectral as frs
import features.root.temporal as frt
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
    warnings.filterwarnings("ignore")
    queries = os.listdir(PATH_QUERIES)

    return queries


def process():
    print("Processing already computed features...")
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
    for dist in TYPES_DISTANCES:
        gen_distances(dist)
        gen_distances(dist, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, default_features)
        gen_distances(dist, OUT_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_DISTANCES)
        gen_distances(dist, OUT_PATH_ALL_ROOT_FEATURES, OUT_PATH_DEFAULT_DISTANCES)


def correlate_features():
    librosa_sim_matrix = np.genfromtxt(OUT_PATH_ALL_TEST_FEATURES, delimiter=DELIMITER_FEATURE)
    root_sim_matrix = np.genfromtxt(OUT_PATH_ALL_ROOT_TEST_FEATURES, delimiter=DELIMITER_FEATURE)

    import pandas as pd
    df1 = pd.DataFrame(librosa_sim_matrix)
    df2 = pd.DataFrame(root_sim_matrix)
    l = len(librosa_sim_matrix)
    mean = int()

    for i in range(l):
        mean += df1[i].corr(df2[i])

    print(mean / l)


def analyse_similarity(queries):
    for query in queries:
        results_obj = objective_analysis(query=query)
        results_obj_ids = list(map(lambda x: x[0], results_obj))
        print("[DEBUG] Ranking results for query %s based on metadata" % query)

        for i in range(len(results_obj)):
            curr = results_obj[i]
            print("%d - %s: %d" % (i + 1, curr[0], curr[3]))

        for dist in TYPES_DISTANCES:
            print("[DEBUG] Ranking results for query %s based on '%s' distanced features" % (query, dist))
            results_features, dist = rank_by_sim_analysis(query, OUT_PATH_DEFAULT_DISTANCES + dist + EXTENSION_CSV, IN_DIR_PATH_ALL_DATABASE)
            results_features_ids = list()
            for i in range(len(results_features)):
                results_features_ids.append(results_features[i].strip(EXTENSION_MP3))
                print("%d - %s (%.4f)" % (i + 1, results_features[i], dist[i]))
            print("Precision: %.2f" % calc_precision(results_features_ids[1:], results_obj_ids))


def main():
    """
        Main function.
    """

    queries = setup()
    objective_analysis()
    default_features = process()
    generate_distances(default_features)
    analyse_similarity(queries)


if __name__ == '__main__':
    main()
