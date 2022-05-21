# region Dependencies


import numpy as np
import librosa.feature as lf


# endregion Dependencies


# region Const

DEFAULT_N_MFCC = 13


# endregion Const


# region Public Functions


def calc_mfcc(audio_buffer: np.ndarray,
              n: int = DEFAULT_N_MFCC) -> np.ndarray:
    """
    Function used to calculate the MFFCs per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the MFFCs will be extracted.
    :param n: The number of MFCCs to return (default: 13).
    :return: The calculated MFFCs.
    """
    return lf.mfcc(y=audio_buffer, n_mfcc=n)


def calc_centroid(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the spectral centroid per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the spectral centroid will be extracted.
    :return: The calculated spectral centroid.
    """
    return lf.spectral_centroid(y=audio_buffer)


def calc_bandwidth(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the spectral bandwidth per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the spectral bandwidth will be extracted.
    :return: The calculated spectral bandwidth.
    """
    return lf.spectral_bandwidth(y=audio_buffer)


def calc_contrast(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the spectral contrast per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the spectral contrast will be extracted.
    :return: The calculated spectral contrast.
    """
    return lf.spectral_contrast(y=audio_buffer)


def calc_flatness(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the spectral flatness per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the spectral flatness will be extracted.
    :return: The calculated spectral flatness.
    """
    return lf.spectral_flatness(y=audio_buffer)


def calc_roll_off(audio_buffer) -> np.ndarray:
    """
    Function used to calculate the spectral roll off per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The audio buffer from which the spectral roll off will be extracted.
    :return: The calculated spectral roll off.
    """
    return lf.spectral_rolloff(audio_buffer)


# endregion Public Functions
