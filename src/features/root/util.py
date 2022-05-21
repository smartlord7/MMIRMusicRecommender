# region Dependencies

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import get_window


# endregion Dependencies


# region Public Functions


def normalize(array: np.ndarray) -> np.ndarray:
    """
    Function used to normalize the array based on its max value.
    :param array: The array to normalize.
    :return: The normalized array.
    """
    array = array / np.max(np.abs(array))

    return array


def frame(array: np.ndarray,
          win_length: float = 2048,
          hop_size: float = 23.22,
          sr: float = 22050) -> np.ndarray:
    """
    Function used to compute the windows of a given array based on the sliding window method.
    :param array: The array from which the windows will be extracted.
    :param win_length: The size of the window used when applying the sliding window method to obtain the audio_buffer frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the audio_buffer frames (default: 23.22ms).
    :param sr: the sample rate.
    :return: The calculated frames.
    """
    array = np.pad(array, int(win_length / 2), mode='reflect')
    len_frame = np.round(sr * hop_size / 1000).astype(np.int32)
    n_frames = int((len(array) - win_length) / len_frame) + 1
    frames = np.zeros((n_frames, win_length))

    for n in range(n_frames):
        frames[n] = array[n * len_frame:n * len_frame + win_length]

    return frames


def windowed_frame(array: np.ndarray,
                   win_type: str = "hann",
                   win_length: int = 2048,
                   hop_size: float = 23.22,
                   sr: float = 22050) -> np.ndarray:
    """
    Function used to compute the windows of a given array based on the sliding window method with a custom window.
    :param array: The array from which the windows will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the audio_buffer frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the audio_buffer frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral centroid.
    """
    framed = frame(array, win_length, hop_size, sr)
    window = get_window(win_type, win_length, fftbins=True)
    framed_w_window = (framed * window)

    return framed_w_window


def power(array) -> np.ndarray:
    """
    Function used to calculate the power of a given array.
    :param array: The array from which the power will be extracted.
    :return: The calculated power.
    """
    return np.square(np.abs(array))


def freq_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """
    Function used to transpose a set of frequencies to the Mel domain.
    :param frequencies: The frequencies to transpose.
    :return: The Mel-transposed frequencies.
    """
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def met_to_freq(mel_values: np.ndarray) -> np.ndarray:
    """
    Function used to transpose a set of Mel coefficients to the frequency domain.
    :param mel_values: The Mel coefficients to transpose.
    :return: The transposed frequencies.
    """
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def mel_filter_points(f_min: np.ndarray,
                      f_max: np.ndarray,
                      n: int,
                      win_length: int = 2048,
                      sr: float = 22050) -> tuple:
    """
    Function used to calculate the mel filter points given a certain configuration of the desired Mel filters.
    :param f_min:
    :param f_max:
    :param n: The number of filter points to calculate.
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated Mel filter points.
    """
    f_min = freq_to_mel(f_min)
    f_max = freq_to_mel(f_max)

    mel = np.linspace(f_min, f_max, num=n + 2)
    frequencies = met_to_freq(mel)

    return np.floor((win_length + 1) / sr * frequencies).astype(np.int64), frequencies


def mel_filter_bank(filter_points: np.ndarray,
                    win_length: int = 2048,
                    debug: bool = False) -> np.array:
    """
    Function used to calculate a bank (set) of Mel filters.
    :param filter_points: The number of filter points to use.
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param debug: Specifies if plots related with process should be presented.
    :return:The calculated bank of Mel filters.
    """
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


def dct(n: int,
        length: int) -> np.array:
    """
    Function used to generate the Discrete Cosine Transform for a discrete dataset of a certain length.
    :param n: The period, in samples, of the dataset.
    :param length: The length, in samples, of the dataset.
    :return: The calculated DCT values.
    """
    dct_values = np.empty((n, length))
    dct_values[0, :] = 1.0 / np.sqrt(length)

    samples = np.arange(1, 2 * length, 2) * np.pi / (2.0 * length)

    for i in range(1, n):
        dct_values[i, :] = np.cos(i * samples) * np.sqrt(2.0 / length)

    return dct_values


def parabolic(f: np.array,
              x: np.array):
    """
    Function used to calculate a quadratic interpolation of a discrete series with a given domain and counter-domain.
    :param f: The series counter-domain.
    :param x: The series domain.
    :return: The calculated quadratic interpolation.
    """
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)

    return xv, yv


# endregion Public Functions
