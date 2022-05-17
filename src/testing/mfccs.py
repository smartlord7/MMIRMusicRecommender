import sounddevice as sd
from scipy.io import wavfile
from testing.spectral import calc_mfcc


def main():
    path = "C:/Users/ssimoes/Downloads/transferir.wav"
    sr, audio = wavfile.read(path)
    sd.play(audio, sr)
    x = calc_mfcc(audio, sr=22050, hop_size=23.22, debug=True)


if __name__ == "__main__":
    main()
