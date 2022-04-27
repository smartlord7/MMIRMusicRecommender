import os
import warnings
from util.process import *
from features.temporal import *


def process_data(process_callback,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_FEATURES,
                 in_extension=EXTENSION_DATA):

    with open(out_path, "w") as out_file:
        for data_file_name in os.listdir(dir_path):
            if data_file_name.endswith(in_extension):
                print("Processing %s..." % data_file_name)
                data, _ = librosa.load(dir_path + data_file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)
                processed = process_callback(data)
                np.savetxt(out_file, processed)
                out_file.write(FEATURE_DELIM)


def main():
    """
    Main function.
    """

    warnings.filterwarnings("ignore")
    process_data(featurize)


if __name__ == '__main__':
    main()
