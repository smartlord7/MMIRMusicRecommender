import warnings

import librosa as librosa
import librosa.display
import librosa.beat
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np


def test1():

    # --- Load file
    # fName = "--/Queries/MT0000414517.mp3"
    fName = "data/queries/MT0000202045.mp3"
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono=mono)
    print(y.shape)
    print(fs)

    # --- Play Sound
    sd.play(y, sr, blocking=False)

    # --- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)

    # --- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # --- Extract features
    rms = librosa.feature.rms(y=y)
    rms = rms[0, :]
    print(rms.shape)
    times = librosa.times_like(rms)
    plt.figure(), plt.plot(times, rms)
    plt.xlabel('Time (s)')
    plt.title('RMS')

    plt.show()