import librosa.feature as lf


def mfcc(matrix, n):
    return lf.mfcc(matrix, n_mfcc=n)


def centroid(matrix):
    return lf.spectral_centroid(matrix)


def bandwidth(matrix, n_bands):
    return lf.spectral_bandwidth(matrix, n_bands)


def contrast(matrix):
    return lf.spectral_contrast(matrix)


def flatness(matrix):
    return lf.spectral_flatness(matrix)


def rolloff(matrix):
    return lf.spectral_rolloff(matrix)


