import librosa
import librosa.feature as lf


def fundamental_freq(matrix, fmin, fmax):
    return librosa.yin(matrix, fmin=fmin, fmax=fmax)


def rms(matrix):
    return lf.rms(matrix)


def zero_crossing_rate(matrix):
    return lf.zero_crossing_rate(matrix)
