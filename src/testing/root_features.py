import sounddevice as sd
from scipy.io import wavfile
from testing.spectral import calc_mfcc, calc_centroid, calc_bandwidth, calc_flatness, calc_roll_off
from testing.temporal import calc_zero_crossing_rate


def main():
    path = "C:/Users/ssimoes/Downloads/transferir.wav"
    sr, audio = wavfile.read(path)
    sd.play(audio, sr)
    mfcc = calc_mfcc(audio)
    centroid = calc_centroid(audio)
    bandwidth = calc_bandwidth(audio)
    flatness = calc_flatness(audio)
    roll_off = calc_roll_off(audio)
    zero_crossing_rate = calc_zero_crossing_rate(audio)


if __name__ == "__main__":
    main()
