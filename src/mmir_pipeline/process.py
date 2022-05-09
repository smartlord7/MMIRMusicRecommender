import os
from os.path import isfile

import numpy as np
from sklearn import preprocessing
from const import *
from features.librosa_wrap.misc import *
from features.librosa_wrap.spectral import *
from features.librosa_wrap.temporal import *
from scipy import stats as stats


def normalize_min_max(matrix):
    """
    Given one matrix,this function will normalize it within the min and max values..
    :param matrix: The used matrix.
    :return: normalized matrix.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)

    return matrix


def statify(data):
    """
    Given a feature,calculates its statistics.
    """
    shp = data.shape
    axis = len(shp) - 1

    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    skewness = stats.skew(data, axis=axis)
    kurtosis = stats.kurtosis(data, axis=axis)
    median = np.median(data, axis=axis)
    max_val = np.max(data, axis=axis)
    min_val = np.min(data, axis=axis)

    if axis == 0:
        array = np.array([mean, std, skewness, kurtosis, median, max_val, min_val])
    else:
        array = np.empty((data.shape[0], 7))
        for i in range(data.shape[0]):
            array[i, :] = [mean[i], std[i], skewness[i], kurtosis[i], median[i], max_val[i], min_val[i]]

    return array.flatten()


def featurize(data):
    """
    Function that will calculate the features.
    """
    mfcc = statify(calc_mfcc(data, N_MFCC))
    spectral_centroid = statify(calc_centroid(data))
    spectral_bandwidth = statify(calc_bandwidth(data))
    spectral_contrast = statify(calc_contrast(data))
    spectral_flatness = statify(calc_flatness(data))
    spectral_rolloff = statify(calc_rolloff(data))

    fundamental_frequency = statify(calc_fundamental_freq(data, MIN_YIN_FREQUENCY, MAX_YIN_FREQUENCY))
    rms = statify(calc_rms(data))
    zero_crossing_rate = statify(calc_zero_crossing_rate(data))

    tempo = calc_tempo(data)

    features = np.concatenate((mfcc,
                               spectral_centroid,
                               spectral_bandwidth,
                               spectral_contrast,
                               spectral_flatness,
                               spectral_rolloff,
                               fundamental_frequency,
                               rms,
                               zero_crossing_rate,
                               tempo))

    return features


def process_data(process_callback,
                 dir_path=IN_DIR_PATH_ALL_DATABASE,
                 out_path=OUT_PATH_ALL_FEATURES,
                 in_extension=EXTENSION_MP3):
    """
    Function used to process the given data.
    """

    if isfile(out_path):
        return

    data_files = os.listdir(dir_path)
    data_files.sort()
    all_processed = np.empty((len(data_files), N_COLS))
    i = int()

    for data_file_name in data_files:
        if data_file_name.endswith(in_extension):
            print("[DEBUG] Processing %s..." % data_file_name)
            data = librosa.load(dir_path + data_file_name, sr=SAMPLING_RATE, mono=IS_AUDIO_MODE_MONO)[0]
            processed = process_callback(data)
            all_processed[i] = processed
            i += 1

    all_processed = normalize_min_max(all_processed)
    np.savetxt(out_path, all_processed, fmt='%f', delimiter=DELIMITER_FEATURE)