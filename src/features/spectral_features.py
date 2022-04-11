import librosa.feature as lf


def mfcc(matrix, n):
    """
    Function used to calculate the MFCC.
    :param matrix: the given matrix.
    :param n: number of MFCCs to return.
    :return: the MFFC(s).
    """
    return lf.mfcc(matrix, n_mfcc=n)


def centroid(matrix):
    """
    Function used to calculate the centroid.
    :param matrix: the given matrix.
    :return: the spectral centroid.
    """
    return lf.spectral_centroid(matrix)


def bandwidth(matrix, n_bands):
    """
    Function used to calculate the bandwidth.
    :param matrix: the given matrix.
    :param n_bands: the number of bands.
    :return: the spectral bandwidth.
    """
    return lf.spectral_bandwidth(matrix, n_bands)


def contrast(matrix):
    """
    Function used to calculate the contrast.
    :param matrix: the given matrix.
    :return: the spectral contrast.
    """
    return lf.spectral_contrast(matrix)


def flatness(matrix):
    """
    Function used to calculate the flatness.
    :param matrix: the given matrix.
    :return: the spectral flatness.
    """
    return lf.spectral_flatness(matrix)


def rolloff(matrix):
    """
    Function used to calculate the rolloff.
    :param matrix: the given matrix.
    :return: the spectral rolloff.
    """
    return lf.spectral_rolloff(matrix)


