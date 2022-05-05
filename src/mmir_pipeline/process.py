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
