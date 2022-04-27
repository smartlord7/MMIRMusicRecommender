def min_max_normalize(matrix, a=0, b=1):
    """
    Given one matrix,this function will normalize it within the min and max values..
    :param matrix: The used matrix.
    :param a: Min value
    :param b: Max value
    :return: normalized matrix.
    """
    min_val = matrix.min()
    max_val = matrix.max()

    matrix = (a + (matrix - min_val) * (b - a)) / (max_val - min_val)

    return matrix
