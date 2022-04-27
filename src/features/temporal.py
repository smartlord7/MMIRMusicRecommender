import librosa
import librosa.feature as lf


def fundamental_freq(matrix, fmin, fmax):
    """
    Function used to get the fundamental frequency.
    :param matrix: the given matrix.
    :param fmin: minimum frequency.
    :param fmax: maximum frequency.
    :return: fundamental frequency.
    """
    return librosa.yin(matrix, fmin=fmin, fmax=fmax)


def rms(matrix):
    """
    Function used to calculate the RMS.
    :param matrix: the given matrix.
    :return: the RMS.
    """

    # Time-Series input
    rms = librosa.feature.rms(y=matrix)

    # Spectrogram input
    s, phase = librosa.magphase(librosa.stft(matrix))
    rms2 = librosa.feature.rms(S=s)

    '''
    # Ainda não consegui testar
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(s, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    '''

    return rms


def zero_crossing_rate(matrix):
    """
    Function used to calculate the zero crossing rate.
    :param matrix: the given matrix.
    :return: the zero crossing rate.
    """
    return lf.zero_crossing_rate(matrix)
