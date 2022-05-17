import scipy.fftpack
from features.root.util import *


def calc_mfcc(data: np.ndarray,
              win_type: str = "hann",
              win_length: int = 2048,
              hop_size: float = 23.22,
              sr: float = 22050,
              debug: bool = False):

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

    framed_w_window = windowed_frame(data, win_type, win_length, hop_size, sr)
    magnitudes = np.abs(np.fft.fft(framed_w_window))
    freq = np.fft.fftfreq(win_length)[None, ...]
    centroids = np.sum(freq * magnitudes, axis=1) / np.sum(magnitudes, axis=1)

    return centroids


def calc_bandwidth():
    pass


def calc_contrast():
    pass


def calc_flatness():
    pass


def calc_rolloff():
    pass
