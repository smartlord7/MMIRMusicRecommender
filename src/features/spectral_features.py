import librosa.feature as lf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


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

    # Time-Series input
    cent = librosa.feature.spectral_centroid(y=matrix, sr=22050)

    # Spectrogram input
    s, phase = librosa.magphase(librosa.stft(y=matrix))
    cent = librosa.feature.spectral_centroid(S=s)


    '''
    # Attempt to plot, but doesn't work :(
    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(matrix)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=22050, ax=ax)
    ax.plot(times, cent.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram')
    '''

    return cent


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

    # Time-Series input
    flat = librosa.feature.spectral_flatness(y=matrix)

    # Spectrogram input
    s, phase = librosa.magphase(librosa.stft(y=matrix))
    flat = librosa.feature.spectral_flatness(S=s)

    # Power Spectrogram input
    s_power = s ** 2
    flat = librosa.feature.spectral_flatness(S=s_power, power=1.0)

    return flat


def rolloff(matrix):
    """
    Function used to calculate the rolloff.
    :param matrix: the given matrix.
    :return: the spectral rolloff.
    """
    return lf.spectral_rolloff(matrix)


