FEATURE_DELIM = ","
IN_PATH_DEFAULT_FEATURES = "../data/features/default/top100_features.csv"
OUT_PATH_DEFAULT_FEATURES = "../out/features/default/top100_features_processed.csv"
IN_DIR_PATH_ALL_DATABASE = "../data/database/all/"
OUT_PATH_ALL_FEATURES = "../out/features/calculated/all.csv"
OUT_PATH_DISTANCES = "../out/distances/"
OUT_PATH_DEFAULT_DISTANCES = "../out/distances/default/"
EXTENSION_MP3 = ".mp3"
EXTENSION_CSV = ".csv"

TYPES_DISTANCES = ["euclidean", "manhattan", "cosine"]

SAMPLING_RATE = 22500
IS_AUDIO_MODE_MONO = True

N_MFCC = 13
MIN_YIN_FREQUENCY = 20
MAX_YIN_FREQUENCY = SAMPLING_RATE / 2

N_COLS = 190
