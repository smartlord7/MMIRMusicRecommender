# region Dependencies


import librosa
import librosa.feature as lf


# endregion Dependencies


# region Const
import numpy as np

DEFAULT_SAMPLING_RATE = 22500
DEFAULT_MIN_YIN_FREQUENCY = 20
DEFAULT_MAX_YIN_FREQUENCY = DEFAULT_SAMPLING_RATE / 2


# endregion Const


# region Public Functions


def calc_fundamental_freq(audio_buffer: np.ndarray,
                          f_min: float = DEFAULT_MIN_YIN_FREQUENCY,
                          f_max: float = DEFAULT_MAX_YIN_FREQUENCY) -> np.ndarray:
    """
    Function used to calculate the fundamental frequency (f0) per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The buffer from which the fundamental frequency will be extracted.
    :param f_min: A lower bound for the estimated fundamental frequency.
    :param f_max: An upper bound for the estimated fundamental frequency.
    :return: The calculated fundamental frequency.
    """

    f = librosa.yin(y=audio_buffer, fmin=f_min, fmax=f_max)
    f[f == fmax] = 0

    return f


def calc_rms(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the root mean squared per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The buffer from which the RMS will be extracted.
    :return: The calculated RMS.
    """

    return lf.rms(y=audio_buffer)


def calc_zero_crossing_rate(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the zero crossing rate per window of a given audio buffer, using the librosa library.
    :param matrix: The buffer from which the zero crossing rate will be extracted.
    :return: The calculated zero crossing rate.
    """
    return lf.zero_crossing_rate(y=audio_buffer)


# region Public Functions
