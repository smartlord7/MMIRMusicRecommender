import scipy.fftpack
from util import *


def calc_mfcc(data: np.ndarray,
              win_type: str = "hann",
              win_length: int = 2048,
              hop_size: float = 23.22,
              sr: float = 22050,
              debug: bool = False):
    """
    Function used to calculate the MFCC.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr:
    :param debug: used to debug.
    :return: the MFCC.
    """
    normalized = normalize(data)
    framed_w_window = windowed_frame(normalized, win_type, win_length, hop_size, sr).T
    frames_fft = np.empty((int(1 + win_length // 2), framed_w_window.shape[1]), dtype=np.complex64, order='F')

    for n in range(frames_fft.shape[1]):
        frames_fft[:, n] = scipy.fftpack.fft(framed_w_window[:, n], axis=0)[:frames_fft.shape[0]]

    frames_fft = frames_fft.T
    pwr = power(frames_fft)
    f_min = 0
    f_max = sr / 2
    n_mel_filters = 10

    filter_points, mel_freq = mel_filter_points(f_min, f_max, n_mel_filters, win_length, sr=sr)
    filters = mel_filter_bank(filter_points, win_length, debug)
    norm = 2.0 / (mel_freq[2: n_mel_filters + 2] - mel_freq[: n_mel_filters])
    filters *= norm[:, np.newaxis]
    filtered = np.dot(filters, np.transpose(pwr))

    if debug:
        plt.figure(figsize=(15, 4))
        for n in range(filters.shape[0]):
            plt.plot(filters[n])

    log = 10.0 * np.log10(filtered)

    n_dct_filter = 40
    dct_filters = dct(n_dct_filter, n_mel_filters)
    mfcc = np.dot(dct_filters, log)

    if debug:
        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, len(data) / sr, num=len(data)), data)
        plt.imshow(mfcc, aspect='auto', origin='lower')

    return mfcc


def calc_centroid(data: np.ndarray,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050):
    """
    Function used to calculate the centroid.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr:
    :return: the centroid.
    """
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    freq = np.fft.fftfreq(win_length)[None, ...]
    centroids = np.sum(freq * magnitudes, axis=1) / np.sum(magnitudes, axis=1)

    return centroids


def calc_bandwidth(data: np.ndarray,
                   order: int = 2,
                   win_type: str = "hann",
                   win_length: int = 2048,
                   hop_size: float = 23.22,
                   sr: float = 22050):
    """
    Function used to calculate the bandwidth.
    :param data: is the given data.
    :param order: is the bandwidth order.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr:
    :return: the bandwidth.
    """
    centroids = calc_centroid(data, win_type, win_length, hop_size, sr)
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    frequencies = np.fft.fftfreq(win_length)[None, ...]
    bandwidths = np.empty(framed_w_window.shape[0])

    for i in range(len(framed_w_window)):
        bandwidths[i] = np.sum(magnitudes[i] * (frequencies - centroids[i]) ** order) ** 1 / order

    return bandwidths


def calc_contrast():
    """
    Function used to calculate the contrast.
    :return: the contrast.
    """
    pass


def calc_flatness(data: np.ndarray,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050):
    """
    Function used to calculate the flatness.
    :param data: is the given data.
    :param win_type: is the window type.
    :param win_length: is the window length
    :param hop_size: is the hop size
    :param sr:
    :return: the flatness.
    """
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    l = len(magnitudes)

    geometric_mean = np.exp(1 / l * np.sum(np.log(magnitudes), axis=1))
    arithmetic_mean = 1 / l * np.sum(magnitudes, axis=1)

    flatness = 20 * np.log10(geometric_mean / arithmetic_mean)

    return flatness


def calc_roll_off(data: np.ndarray,
                  roll_perc: float = 0.85,
                  win_type: str = "hann",
                  win_length: int = 2048,
                  hop_size: float = 23.22,
                  sr: float = 22050):
    """
    Function used to calculate the roll off.
    :param data: is the given data.
    :param roll_perc: is the roll percentage.
    :param win_type: is the window type.
    :param win_length: is the window length.
    :param hop_size: is the hop size.
    :param sr:
    :return: the roll off.
    """
    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    frequencies = np.fft.fftfreq(win_length)
    total_energy = np.cumsum(magnitudes, axis=-2)
    threshold = roll_perc * total_energy[..., -1, :]
    threshold = np.expand_dims(threshold, axis=-2)
    ind = np.where(total_energy < threshold, np.nan, 1)
    roll_off = np.nanmin(ind * frequencies, axis=-2, keepdims=True)

    return roll_off
