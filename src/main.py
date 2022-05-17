import warnings

from scipy.stats import stats
from features.librosa_wrap.misc import *
from features.librosa_wrap.spectral import *
from features.librosa_wrap.temporal import *
from mmir_pipeline.process import *
from testing.features import *
from mmir_pipeline.sim_analysis import gen_distances, rank_by_sim_analysis, objective_analysis, \
    calc_precision


FUNCTIONS_STATISTICS = [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.max, np.min]
FUNCTIONS_FEATURES = [calc_mfcc, calc_centroid, calc_bandwidth, calc_contrast,
                      calc_flatness, calc_rolloff, calc_fundamental_freq, calc_rms,
                      calc_zero_crossing_rate, calc_tempo]


def setup():
    warnings.filterwarnings("ignore")
    queries = os.listdir(PATH_QUERIES)

    return queries


def process():
    default_features = process_default_features(IN_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_FEATURES)
    process_data(FUNCTIONS_STATISTICS, FUNCTIONS_FEATURES)

    return default_features


def generate_distances(default_features):
    for dist in TYPES_DISTANCES:
        gen_distances(dist)
        gen_distances(dist, OUT_PATH_ALL_FEATURES, OUT_PATH_DISTANCES, default_features)
        gen_distances(dist, OUT_PATH_DEFAULT_FEATURES, OUT_PATH_DEFAULT_DISTANCES, default_features)


def analyse_similarity(queries):
    for query in queries:
        results_obj = objective_analysis(query=query)
        results_obj_ids = list(map(lambda x: x[0], results_obj))
        print("[DEBUG] Ranking results for query %s based on metadata" % query)

        for i in range(len(results_obj)):
            curr = results_obj[i]
            print("%d - %s: %s by %s - Points %d" % (i + 1, curr[0], curr[1], curr[2], curr[3]))

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
