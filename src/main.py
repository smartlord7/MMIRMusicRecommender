import os
import warnings
from util.process import *
from features.temporal import *


def process_data(dir_path, extension, process_callback):
    data = dict()
    for file_name in os.listdir(dir_path):
        if file_name.endswith(extension):
            print("Processing %s..." % file_name)
            matrix, _ = librosa.load(dir_path + "/" + file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)
            data[file_name] = process_callback(matrix)

    return data


def main():
    """
    Main function.
    """

    warnings.filterwarnings("ignore")
    data = process_data(IN_DIR_PATH_ALL_DATABASE, EXTENSION_DATA, featurize)


if __name__ == '__main__':
    main()
