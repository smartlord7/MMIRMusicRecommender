# region Dependencies


import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
from features.root.util import normalize, windowed_frame, mel_filter_points, mel_filter_bank, dct, power


# endregion Dependencies


# region Const


F_MIN = 0
N_MEL_FILTERS = 10
MEL_NORM_FACTOR = 2.0


# endregion Const


# region Public Functions


def calc_mfcc(audio_buffer: np.ndarray,
              win_type: str = "hann",
              win_length: int = 2048,
              hop_size: float = 23.22,
              sr: float = 22050,
              n_mfcc: int = 13,
              debug: bool = False) -> np.ndarray:
    """
    Function used to calculate the MFCCs per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the data frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :param n_mfcc: The number of MFFCs to compute (default: 13).
    :param debug: Specifies if plots related with process should be presented.
    :return: The calculated MFCCs.
    """
    normalized = normalize(audio_buffer)
    framed_w_window = windowed_frame(normalized, win_type, win_length, hop_size, sr).T
    frames_fft = np.empty((int(1 + win_length // 2), framed_w_window.shape[1]), dtype=np.complex64, order='F')

    for n in range(frames_fft.shape[1]):
        frames_fft[:, n] = scipy.fftpack.fft(framed_w_window[:, n], axis=0)[:frames_fft.shape[0]]

    frames_fft = frames_fft.T
    pwr = power(frames_fft)

    filter_points, mel_freq = mel_filter_points(F_MIN, f_max, N_MEL_FILTERS, win_length, sr=sr)
    filters = mel_filter_bank(filter_points, win_length, debug)
    norm = MEL_NORM_FACTOR / (mel_freq[2: N_MEL_FILTERS + 2] - mel_freq[: n_mel_filters])
    filters *= norm[:, np.newaxis]
    filtered = np.dot(filters, np.transpose(pwr))

    if debug:
        plt.figure(figsize=(15, 4))
        for n in range(filters.shape[0]):
            plt.plot(filters[n])

    log = 10.0 * np.log10(filtered)

    dct_filters = dct(n_mfcc, N_MEL_FILTERS)
    mfcc = np.dot(dct_filters, log)

    if debug:
        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, len(audio_buffer) / sr, num=len(audio_buffer)), audio_buffer)
        plt.imshow(mfcc, aspect='auto', origin='lower')

    return mfcc


def calc_centroid(audio_buffer: np.ndarray,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the spectral centroid per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the data frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral centroid.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    freq = np.fft.fftfreq(win_length)[None, ...]
    centroids = np.sum(freq * magnitudes, axis=1) / np.sum(magnitudes, axis=1)

    return centroids


def calc_bandwidth(audio_buffer: np.ndarray,
                   order: int = 2,
                   win_type: str = "hann",
                   win_length: int = 2048,
                   hop_size: float = 23.22,
                   sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the spectral bandwidth per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param order: The exponent used to calculate the spectral bandwidth.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the data frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral bandwidth.
    """
    centroids = calc_centroid(audio_buffer, win_type, win_length, hop_size, sr)
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    frequencies = np.fft.fftfreq(win_length)[None, ...]
    bandwidths = np.empty(framed_w_window.shape[0])

    for i in range(len(framed_w_window)):
        bandwidths[i] = np.sum(magnitudes[i] * (frequencies - centroids[i]) ** order) ** 1 / order

    return bandwidths


def calc_contrast(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the spectral contrast per window of a given audio buffer, using the librosa library - not implemented.
    :param audio_buffer: The audio buffer from which the spectral contrast will be extracted.
    """
    pass


def calc_flatness(audio_buffer: np.ndarray,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the spectral flatness per window of a given audio buffer, using a root-implemented logic.
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the data frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral flatness centroid.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    l = len(magnitudes)

    geometric_mean = np.exp(1 / l * np.sum(np.log(magnitudes), axis=1))
    arithmetic_mean = 1 / l * np.sum(magnitudes, axis=1)

    flatness = 20 * np.log10(geometric_mean / arithmetic_mean)

    return flatness


def calc_roll_off(audio_buffer: np.ndarray,
                  roll_perc: float = 0.85,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050) -> np.ndarray:
    """
    Function used to calculate the spectral roll off per window of a given audio buffer, using a root-implemented logic.
    :param roll_perc: The percentage of energy of each frame, from which, the lowest frequency will be calculated (default: 0.85)
    :param audio_buffer: The buffer from which the MFCCs will be extracted.
    :param win_type: The window type used when applying the sliding window method (default: "hann").
    :param win_length: The size of the window used when applying the sliding window method to obtain the data frames (default: 2048).
    :param hop_size: The hop size used when applying the sliding window method to obtain the data frames (default: 23.22ms).
    :param sr: The sample rate used when applying discrete transforms (default: 22050).
    :return: The calculated spectral roll off.
    """
    framed_w_window = windowed_frame(audio_buffer, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    frequencies = np.fft.fftfreq(win_length)
    total_energy = np.cumsum(magnitudes, axis=-2)
    threshold = roll_perc * total_energy[..., -1, :]
    threshold = np.expand_dims(threshold, axis=-2)
    ind = np.where(total_energy < threshold, np.nan, 1)
    roll_off = np.nanmin(ind * frequencies, axis=-2, keepdims=True)

    return roll_off


# endregion Public Functions
