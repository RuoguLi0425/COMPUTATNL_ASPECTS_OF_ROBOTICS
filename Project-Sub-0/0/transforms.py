from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """ Check if array is a valid transform.
    You can refer to the lecture notes to
    see how to check if a matrix is a valid
    transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    # TODO:
    if t.shape != (4, 4):
        return False

    R = t[:3, :3]
    Tran = t[:3, 3:].reshape(-1)
    LastRow = t[3:, :4]
    RT = R.T
    RRT = R@RT
    RTR = RT@R
    errorR = RRT-RTR

    condition1 = np.all(np.abs(errorR) < tolerance)
    condition2 = np.all(np.abs(LastRow - [0, 0, 0, 1]) < tolerance)
    condition3 = np.abs(np.linalg.det(R) - 1) < tolerance

    if condition1 and condition2 and condition3:
        return True
    else:
        return False

def transform_concat(t1, t2):
    """ Concatenate two transforms. Hint:
        use numpy matrix multiplication.

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    # TODO:
    if transform_is_valid(t1,tolerance=1e-3):
        if transform_is_valid(t2,tolerance=1e-3):
            return t1@t2
        else:
            raise ValueError('Invalid input transform t2')
    else:
        raise ValueError('Invalid input transform t1')


def transform_point3s(t, ps):
    """
    In:
        t: Numpy array [4, 4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    """
    if transform_is_valid(t) == False:
        raise ValueError('Invalid input transform t')

    if ps.shape[1:] != (3,):
        raise ValueError('Invalid input points p')

    TRP = np.empty((len(ps), 4), dtype=np.float32)
    ones = np.ones((len(ps), 1), dtype=np.float32)
    TRP[:, :3] = ps
    TRP[:, 3:] = ones
    TP = TRP @ t.T
    return TP[:, :3]


def transform_inverse(t):
    """Find the inverse of the transfom. Hint:
        use Numpy's linear algebra native methods.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    # TODO:
    if transform_is_valid(t,tolerance=1e-3):
        T_inverse = np.linalg.inv(t)
        return T_inverse
    else:
        raise ValueError('Invalid input transform t')

