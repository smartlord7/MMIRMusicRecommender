from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import get_window


def normalize(data):
    """
    Function used to normalize the data.
    :param data: the given information-
    :return: the normalized data.
    """
    data = data / np.max(np.abs(data))

    return data


def frame(data,
          win_length: float = 2048,
          hop_size: float = 23.22,
          sr: float = 22050):
    """
    Function used to calculate the frames.
    :param data: the given data.
    :param win_length: the window length.
    :param hop_size: the hop size.
    :param sr:
    :return: the frames
    """
    data = np.pad(data, int(win_length / 2), mode='reflect')
    len_frame = np.round(sr * hop_size / 1000).astype(np.int32)
    n_frames = int((len(data) - win_length) / len_frame) + 1
    frames = np.zeros((n_frames, win_length))

    for n in range(n_frames):
        frames[n] = data[n * len_frame:n * len_frame + win_length]

    return frames


def windowed_frame(data: np.ndarray,
                   win_type: str = "hann",
                   win_length: int = 2048,
                   hop_size: float = 23.22,
                   sr: float = 22050):
    """
    Function used to calculate the windowed frames.
    :param data: is the given data.
    :param win_type: the window type.
    :param win_length: the windowd length.
    :param hop_size: the hop size.
    :param sr:
    :return: the windowed frames.
    """
    framed = frame(data, win_length, hop_size, sr)
    window = get_window(win_type, win_length, fftbins=True)
    framed_w_window = (framed * window)

    return framed_w_window


def power(data):
    """

    """
    return np.square(np.abs(data))


def freq_to_mel(freqs):

    return 2595.0 * np.log10(1.0 + freqs / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filter_points(f_min, f_max, n, win_length=2048, sr=22050):
    f_min = freq_to_mel(f_min)
    f_max = freq_to_mel(f_max)

    mel = np.linspace(f_min, f_max, num=n + 2)
    frequencies = met_to_freq(mel)

    return np.floor((win_length + 1) / sr * frequencies).astype(np.int64), frequencies


def mel_filter_bank(filter_points, win_length=2048, debug=False):
    filter_bank = np.zeros((len(filter_points) - 2, int(win_length / 2 + 1)))

    for n in range(len(filter_points) - 2):
        filter_bank[n, filter_points[n]: filter_points[n + 1]] =\
            np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filter_bank[n, filter_points[n + 1]: filter_points[n + 2]] =\
            np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

    if debug:
        plt.figure(figsize=(15, 4))
        for n in range(filter_bank.shape[0]):
            plt.plot(filter_bank[n])

    return filter_bank


def dct(n, length):
    """
    Function used to calculate the dct.
    :param n:
    :param length:
    :return:
    """
    dct_values = np.empty((n, length))
    dct_values[0, :] = 1.0 / np.sqrt(length)

    samples = np.arange(1, 2 * length, 2) * np.pi / (2.0 * length)

    for i in range(1, n):
        dct_values[i, :] = np.cos(i * samples) * np.sqrt(2.0 / length)

    return dct_values


def parabolic(f, x):
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)

    return xv, yv
