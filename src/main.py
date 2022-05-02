import os
import warnings
from util.process import *
from features.temporal import *
from testing.features import *


def process_data(process_callback,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_FEATURES,
                 in_extension=EXTENSION_DATA):

    data_files = os.listdir(dir_path)
    all_processed = np.zeros((len(data_files), N_COLS))
    i = int()
    for data_file_name in data_files:
        if data_file_name.endswith(in_extension):
            print("Processing %s..." % data_file_name)
            data, _ = librosa.load(dir_path + data_file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)
            processed = process_callback(data)
            all_processed[i] = processed
            i += 1

    all_processed = normalize_min_max(all_processed).astype(np.float32)

    np.savetxt(out_path, all_processed, fmt='%f', delimiter=FEATURE_DELIM)


def main():
    """
    Main function.
    """

    warnings.filterwarnings("ignore")
    process_default_features(IN_PATH_ORIGINAL_FEATURES, OUT_PATH_ORIGINAL_FEATURES)
    process_data(featurize)


if __name__ == '__main__':
    main()
