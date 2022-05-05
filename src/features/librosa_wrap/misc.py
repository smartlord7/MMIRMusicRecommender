import librosa.beat as lb


def calc_tempo(matrix):
    """
    Function used to calculate the tempo.
    :param matrix: the given matrix.
    :return: the tempo.
    """
    return lb.tempo(y=matrix)

