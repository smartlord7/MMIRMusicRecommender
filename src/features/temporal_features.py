import librosa
import librosa.feature as lf


def fundamental_freq(matrix, fmin, fmax):
    """
    Function used to get the fundamental frequency.
    :param matrix: the given matrix.
    :param fmin: minimum frequency.
    :param fmax: maximum frequency.
    :return: fundamental frequency.
    """
    return librosa.yin(matrix, fmin=fmin, fmax=fmax)


def rms(matrix):
    """
    Function used to calculate the RMS.
    :param matrix: the given matrix.
    :return: the RMS.
    """
    return lf.rms(matrix)


def zero_crossing_rate(matrix):
    """
    Function used to calculate the zero crossing rate.
    :param matrix: the given matrix.
    :return: the zero crossing rate.
    """
    return lf.zero_crossing_rate(matrix)
