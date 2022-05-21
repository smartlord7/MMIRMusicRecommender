# region Dependencies


import os
import librosa
import numpy as np
from const import *
from numpy import NaN, inf
from os.path import isfile
from sklearn import preprocessing


# endregion Dependencies


# region Public Functions


def normalize_min_max(matrix: np.ndarray) -> np.ndarray:
    """
    Function that normalizes the columns of a matrix based in the min-max normalization method.
    :param matrix: The matrix to normalize.
    :return: The normalized matrix.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)

    return matrix


def statify(feature: np.ndarray,
            stats_functions: list) -> np.ndarray:
    """
    Function that calculates a set of statistics over a given feature.
    :param feature: The feature to be statified.
    :param stats_functions: The functions used to compute a specific set of statistics of the feature in question.
    :return: The statified feature.
    """
    shp = feature.shape
    axs = len(shp) - 1
    stats_list = [stat_function(feature, axis=axs) for stat_function in stats_functions]

    if axs == 0:
        stat_feature = np.array(stats_list)
    else:
        stat_feature = np.empty((shp[0], len(stats_functions)))
        for i in range(shp[0]):
            row_stats = list()
            for j in range(len(stats_list)):
                row_stats.append(stats_list[j][i])
            stat_feature[i, :] = row_stats

    return stat_feature.flatten()


def featurize(data: np.ndarray,
              stats_functions: list,
              features_functions: list) -> np.ndarray:
    """
    Function that calculates a given set of features over a piece of data.
    :param stats_functions: The functions used to compute a specific set of statistics of the feature in question.
    :param features_functions: The functions used to compute a specific set of features (spectral, temporal, ...) of the data in question.
    :param data: The data to be featurized.
    :return: The featurized data.
    """

    features_array = list()

    for feature_func in features_functions:
        feature_array = feature_func(data)
        shp = feature_array.shape

        if shp[0] > 1 or len(shp) > 1:
            feature_array = statify(feature_array, stats_functions)

        features_array.append(feature_array)

    return np.concatenate(features_array)


def process_data(stats_functions: list,
                 features_functions: list,
                 dir_path: str = IN_DIR_PATH_ALL_DATABASE,
                 out_path: str = OUT_PATH_ALL_FEATURES,
                 in_extension: str = EXTENSION_MP3) -> None:
    """
    Function used to extract and output the statified features of all files with a certain extension in a specified folder
    :param dir_path: The path of the directory that contains the data to be processed.
    :param out_path: The path of the file to which the features' matrix will be written into.
    :param in_extension: The allowed extension of the data files.
    :param stats_functions: The functions used to compute a specific set of statistics of each multidimensional feature.
    :param features_functions:  The functions used to compute a specific set of features (spectral, temporal, ...) of the data in question.
    :return: None
    """

    if isfile(out_path):
        return

    data_files = os.listdir(dir_path)
    data_files.sort()
    all_processed = list()
    i = int()

    for data_file_name in data_files:
        if data_file_name.endswith(in_extension):
            print("[DEBUG] Processing %s..." % data_file_name)
            data = librosa.load(dir_path + data_file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)[0]
            processed = featurize(data, stats_functions, features_functions)
            all_processed.append(processed)
            i += 1

    all_processed = np.array(all_processed)
    all_processed[all_processed == -inf] = 0
    all_processed[all_processed == inf] = 0
    all_processed[all_processed == NaN] = 0
    all_processed[all_processed == 'nan'] = 0
    all_processed = normalize_min_max(np.array(all_processed))

    np.savetxt(out_path, all_processed, fmt='%f', delimiter=DELIMITER_FEATURE)


def process_default_features(in_path, out_path):
    """
    Function used to process the used features.
    :param in_path: the input directory.
    :param out_path: the output directory.
    :return: required values.
    """
    matrix = np.genfromtxt(in_path, delimiter=DELIMITER_FEATURE)
    values = matrix[1:, 1:matrix.shape[1] - 1]

    values = normalize_min_max(values)
    np.savetxt(out_path, values, fmt='%f', delimiter=DELIMITER_FEATURE)

    return values


# endregion Public Functions