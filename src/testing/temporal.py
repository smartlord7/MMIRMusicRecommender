import numpy as np
import scipy.signal
from testing.util import windowed_frame, parabolic


def calc_fundamental_freq(data: np.ndarray,
                          win_type: str = "hann",
                          win_length: int = 2048,
                          hop_size: float = 23.22,
                          sr: float = 22050):
    """
    Function used to calculate the fundamental frequency.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the windows length.
    :param hop_size: is the hop size.
    :param sr: the sample rate.
    :return: the fundamental frequency.
    """

    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    f0 = np.empty((framed_w_window.shape[0]))
    i = int()

    for frame in framed_w_window:
        corr = scipy.signal.convolve(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        start = np.nonzero(d > 0)[0][0]
        peak = np.argmax(corr[start:]) + start
        px, py = parabolic(corr, peak)
        f0[i] = sr / px
        i += 1

    return f0


def calc_rms(data: np.ndarray,
             win_type: str = "hann",
             win_length: int = 2048,
             hop_size: float = 23.22,
             sr: float = 22050):
    """
    Function used to calculate the root mean square.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr: the sample rate.
    :return: the root mean square.
    """
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    squared = np.sum(framed_w_window ** 2, axis=1) ** (1 / 2)
    rms = squared / framed_w_window.shape[1]

    return rms


def calc_zero_crossing_rate(data: np.ndarray,
                            win_type: str = "hann",
                            win_length: int = 2048,
                            hop_size: float = 23.22,
                            sr: float = 22050):
    """
    Function used to calculate the zero crossing rate.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr: the sample rate.
    :return: the zero crossing rate.
    """
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    zero_crossing_rate = np.sum(np.diff(framed_w_window > 0, axis=1), axis=1)

    return zero_crossing_rate
