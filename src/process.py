import numpy as np
from const import *
from features.misc import *
from features.spectral import *
from features.temporal import *
from scipy import stats as stats


def normalize_min_max(matrix, a=0, b=1):
    """
    Given one matrix,this function will normalize it within the min and max values..
    :param matrix: The used matrix.
    :param a: Min value
    :param b: Max value
    :return: normalized matrix.
    """
    min_val = matrix.min()
    max_val = matrix.max()

    matrix = (a + (matrix - min_val) * (b - a)) / (max_val - min_val)

    return matrix


def statify(data):
    shp = data.shape
    axis = len(shp) - 1

    mean = np.mean(data, axis=axis).astype(np.float64)
    std = np.std(data, axis=axis).astype(np.float64)
    skewness = stats.skew(data, axis=axis).astype(np.float64)
    kurtosis = stats.kurtosis(data, axis=axis).astype(np.float64)
    median = np.median(data, axis=axis).astype(np.float64)
    max = np.max(data, axis=axis).astype(np.float64)
    min = np.min(data, axis=axis).astype(np.float64)

    statified_data = np.concatenate((mean, std, skewness, kurtosis, median, max, min))

    return statified_data


def featurize(data):
    mfcc = statify(calc_mfcc(data, N_MFCC))
    spectral_centroid = statify(calc_centroid(data))
    spectral_bandwidth = statify(calc_bandwidth(data))
    spectral_contrast = statify(calc_contrast(data))
    spectral_flatness = statify(calc_flatness(data))
    spectral_rolloff = statify(calc_rolloff(data))

    fundamental_frequency = statify(calc_fundamental_freq(data, 1, SAMPLING_RATE))
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

    features = normalize_min_max(features)

    return features
