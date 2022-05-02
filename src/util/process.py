import numpy as np
from sklearn import preprocessing

from const import *
from features.misc import *
from features.spectral import *
from features.temporal import *
from scipy import stats as stats


def normalize_min_max(matrix):
    """
    Given one matrix,this function will normalize it within the min and max values..
    :param matrix: The used matrix.
    :param a: Min value
    :param b: Max value
    :return: normalized matrix.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)

    return matrix


def statify(data):
    shp = data.shape
    axis = len(shp) - 1

    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    skewness = stats.skew(data, axis=axis)
    kurtosis = stats.kurtosis(data, axis=axis)
    median = np.median(data, axis=axis)
    max = np.max(data, axis=axis)
    min = np.min(data, axis=axis)

    statified_data = np.concatenate((mean,
                                     std,
                                     skewness,
                                     kurtosis,
                                     median,
                                     max,
                                     min)).astype(np.float64)

    return statified_data


def featurize(data):
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
