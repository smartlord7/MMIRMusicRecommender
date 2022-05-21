# region Dependencies


import numpy as np
import librosa.beat as lb


# endregion Dependencies


# region Public Functions


def calc_tempo(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Function used to calculate the tempo per window of a given audio buffer, using the librosa library.
    :param audio_buffer: The buffer from which the tempo will be extracted.
    :return: The tempo of the provided audio.
    """
    return lb.tempo(y=audio_buffer)


# endregion Public Functions
