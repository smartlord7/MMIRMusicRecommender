# region Dependencies


import numpy as np
import pandas as pd
from const import OUT_PATH_ALL_ROOT_DISTANCES, OUT_PATH_DISTANCES, DELIMITER_FEATURE


# endregion Dependencies


# region Public Functions


def correlate(path1: str,
              path2: str):
    """
    Function used to compute the Pearson correlation between two matrices stored in two CSV files.
    :param path1: The path of the CSV file that contains the first matrix.
    :param path2: The path of the CSV file that contains the second matrix.
    :return: The calculated Pearson correlation coefficients between each row of both matrices
    """
    matrix1 = np.genfromtxt(path1, delimiter=DELIMITER_FEATURE)
    matrix2 = np.genfromtxt(path2, delimiter=DELIMITER_FEATURE)

    matrix1[np.isnan(matrix1)] = 0
    matrix2[np.isnan(matrix2)] = 0
    df1 = pd.DataFrame(matrix1)
    df2 = pd.DataFrame(matrix2)
    l = matrix1.shape[1]
    lt = list()

    for i in range(l):
        val = df1[i].corr(df2[i])
        lt.append(val)

    return np.array(lt)


# endregion Public Functions

