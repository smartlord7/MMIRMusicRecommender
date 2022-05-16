import librosa
import librosa.feature as lf

SAMPLING_RATE = 22500
MIN_YIN_FREQUENCY = 20
MAX_YIN_FREQUENCY = SAMPLING_RATE / 2


def calc_fundamental_freq(matrix, fmin=MIN_YIN_FREQUENCY, fmax=MAX_YIN_FREQUENCY):
    """
    Function used to get the fundamental frequency.
    :param matrix: the given matrix.
    :param fmin: minimum frequency.
    :param fmax: maximum frequency.
    :return: fundamental frequency.
    """

    f = librosa.yin(y=matrix, fmin=fmin, fmax=fmax)
    f[f == fmax] = 0

    return f


def calc_rms(matrix):
    """
    Function used to calculate the RMS.
    :param matrix: the given matrix.
    :return: the RMS.
    """

    return lf.rms(y=matrix)


def calc_zero_crossing_rate(matrix):
    """
    Function used to calculate the zero crossing rate.
    :param matrix: the given matrix.
    :return: the zero crossing rate.
    """
    return lf.zero_crossing_rate(y=matrix)
