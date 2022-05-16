import numpy as np
from scipy.stats import stats
from features.librosa_wrap.spectral import *
from features.librosa_wrap.temporal import *
from features.librosa_wrap.misc import *

DELIMITER_FEATURE = ","
DELIMITER_METADATA_SIMILARITY = ","
DELIMITER_METADATA_PROPERTIES = ","
DELIMITER_METADATA_ADJECTIVES = "; "
WRAPPER_METADATA_ADJECTIVES = "\"'"
IN_PATH_DEFAULT_FEATURES = "../data/features/default/top100_features.csv"
OUT_PATH_DEFAULT_FEATURES = "../out/features/default/top100_features_processed.csv"
IN_DIR_PATH_ALL_DATABASE = "../data/database/all/"
IN_PATH_ORIGINAL_FEATURES = "../data/features/default/top100_features.csv"
OUT_PATH_ORIGINAL_FEATURES = "../out/features/default/top100_features_processed.csv"
OUT_PATH_ALL_FEATURES = "../out/features/calculated/all.csv"
OUT_PATH_DISTANCES = "../out/distances/"
OUT_PATH_DEFAULT_DISTANCES = "../out/distances/default/"
OUT_PATH_CONTEXT_SIMILARITY = "../out/metadata/objective_analysis.csv"
PATH_QUERIES = "../data/queries/"
EXTENSION_MP3 = ".mp3"
EXTENSION_CSV = ".csv"
PATH_METADATA = "../data/database_info/panda_dataset_taffc_metadata.csv"

TYPES_DISTANCES = ["euclidean", "manhattan", "cosine"]
FUNCTIONS_STATISTICS = [np.mean, np.std, stats.skew, stats.kurtosis, np.median, np.max, np.min]
FUNCTIONS_FEATURES = [calc_mfcc, calc_centroid, calc_bandwidth, calc_contrast,
                      calc_flatness, calc_rolloff, calc_fundamental_freq, calc_rms,
                      calc_zero_crossing_rate, calc_tempo]

SAMPLING_RATE = 22500
IS_AUDIO_MODE_MONO = True

N_FEATURE_ARRAY_COLS = 190

COL_METADATA_MUSIC_ID = 0
COL_METADATA_ARTIST = 1
COL_METADATA_QUADRANT = 3
COL_METADATA_EMOTIONS = 9
COL_METADATA_GENRE = 11
