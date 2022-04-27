import librosa.beat as lb


def tempo(matrix):
    """
    Function used to calculate the tempo.
    :param matrix: the given matrix.
    :return: the tempo.
    """
    return lb.tempo(matrix)

