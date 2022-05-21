# region Dependencies


import numpy as np
import scipy.signal
from features.root.util import windowed_frame, parabolic


# endregion Dependencies


# region Public Functions


def calc_fundamental_freq(audio_buffer: np.ndarray,
                          win_type: str = "hann",
                          win_length: int = 2048,
                          hop_size: float = 23.22,
                          sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the fundamental frequency per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the audio_buffer frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the audio_buffer frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral centroid.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    f0 = np.empty((framed_w_window.shape[0]))
    i = int()

    for frame in framed_w_window:
        corr = scipy.signal.convolve(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        if len(d.shape) < 2:
            start = 0
        else:
            start = np.nonzero(d > 0)[0][0]
        peak = np.argmax(corr[start:]) + start
        px, py = parabolic(corr, peak)
        f0[i] = sr / px
        i += 1

    return f0


def calc_rms(audio_buffer: np.ndarray,
             win_type: str = "hann",
             win_length: int = 2048,
             hop_size: float = 23.22,
             sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the root mean squared per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the audio_buffer frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the audio_buffer frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The root mean squared.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    squared = np.sum(framed_w_window ** 2, axis=1) ** (1 / 2)
    rms = squared / framed_w_window.shape[1]

    return rms


def calc_zero_crossing_rate(audio_buffer: np.ndarray,
                            win_type: str = "hann",
                            win_length: int = 2048,
                            hop_size: float = 23.22,
                            sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the zero crossing rate per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the audio_buffer frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the audio_buffer frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The zero crossing rate.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    zero_crossing_rate = np.sum(np.diff(framed_w_window > 0, axis=1), axis=1)

    return zero_crossing_rate


# endregion Public Functions
