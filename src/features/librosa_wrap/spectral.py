import librosa.feature as lf


def calc_mfcc(matrix, n):
    """
    Function used to calculate the MFCC.
    :param matrix: the given matrix.
    :param n: number of MFCCs to return.
    :return: the MFFC(s).
    """
    return lf.mfcc(y=matrix, n_mfcc=n)


def calc_centroid(matrix):
    """
    Function used to calculate the centroid.
    :param matrix: the given matrix.
    :return: the spectral centroid.
    """

    return lf.spectral_centroid(y=matrix)


def calc_bandwidth(matrix):
    """
    Function used to calculate the bandwidth.
    :param matrix: the given matrix.
    :return: the spectral bandwidth.
    """
    return lf.spectral_bandwidth(y=matrix)


def calc_contrast(matrix):
    """
    Function used to calculate the contrast.
    :param matrix: the given matrix.
    :return: the spectral contrast.
    """
    return lf.spectral_contrast(y=matrix)


def calc_flatness(matrix):
    """
    Function used to calculate the flatness.
    :param matrix: the given matrix.
    :return: the spectral flatness.
    """

    return lf.spectral_flatness(y=matrix)


def calc_rolloff(matrix):
    """
    Function used to calculate the rolloff.
    :param matrix: the given matrix.
    :return: the spectral rolloff.
    """
    return lf.spectral_rolloff(matrix)
