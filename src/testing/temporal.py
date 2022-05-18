import numpy as np

from testing.util import windowed_frame


def calc_fundamental_freq():
    pass


def calc_rms(data: np.ndarray,
             win_type: str = "hann",
             win_length: int = 2048,
             hop_size: float = 23.22,
             sr: float = 22050):
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    squared = np.sum(framed_w_window ** 2, axis=1) ** (1 / 2)
    rms = squared / framed_w_window.shape[1]

    return rms


def calc_zero_crossing_rate(data: np.ndarray,
                            win_type: str = "hann",
                            win_length: int = 2048,
                            hop_size: float = 23.22,
                            sr: float = 22050):

    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    zero_crossing_rate = np.sum(np.diff(framed_w_window > 0, axis=1), axis=1)

    return zero_crossing_rate
