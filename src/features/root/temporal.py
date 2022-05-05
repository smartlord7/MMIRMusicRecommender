import numpy as np


def calc_fundamental_freq():
    pass


def calc_rms():
    pass


def calc_zero_crossing_rate(data):
    return np.nonzero(np.diff(data > 0))[0]
