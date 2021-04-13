from numba import jit
import numpy as np

@jit(nopython=True, nogil=True, cache=True)
def bilinear_interpolation(ix, x, y, data_def, data):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    ds_ix = data_def[ix]
    x_start_ix = ds_ix + 6

    x_len = data[ds_ix + 0]

    x_min = data[ds_ix + 2]

    x_max = data[ds_ix + 3]

    y_len = data[ds_ix + 1]

    y_min = data[ds_ix + 4]
    y_max = data[ds_ix + 5]

    y_start_ix = x_start_ix + np.int64(x_len)
    if (y_start_ix) < ds_ix:
        error_code = 3

    if x_max == x_min:
        x_norm = x
    else:
        x_norm = (x - x_min) / (x_max - x_min)

    if y_max == y_min:
        y_norm = y
    else:
        y_norm = (y - y_min) / (y_max - y_min)

    x_frac_ix = x_norm * (x_len - 1)

    x1_ix = np.int64(np.floor(x_frac_ix))

    if x1_ix < 0:
        x1_ix = 0
    elif x1_ix >= x_len:
        x1_ix = np.int64(x_len) - 1

    x1 = data[x1_ix + x_start_ix]

    x2_ix = np.int64(np.ceil(x_frac_ix))

    if x2_ix < 0:
        x2_ix = 0
    elif x2_ix >= x_len:
        x2_ix = np.int64(x_len) - 1

    x2 = data[x2_ix + x_start_ix]

    y_frac_ix = y_norm * (y_len - 1)

    y1_ix = np.int64(np.floor(y_frac_ix))

    if y1_ix < 0:
        y1_ix = 0
    elif y1_ix >= y_len:
        y1_ix = np.int64(y_len) - 1

    y1 = data[y1_ix + y_start_ix]

    y2_ix = np.int64(np.ceil(y_frac_ix))

    if y2_ix < 0:
        y2_ix = 0
    elif y2_ix >= y_len:
        y2_ix = np.int64(y_len) - 1

    y2 = data[y2_ix + y_start_ix]

    q_start_ix = y_start_ix + np.int64(y_len)

    q11 = data[q_start_ix + x1_ix + np.int64(x_len) * y1_ix]
    q12 = data[q_start_ix + x1_ix + np.int64(x_len) * y2_ix]
    q21 = data[q_start_ix + x2_ix + np.int64(x_len) * y1_ix]
    q22 = data[q_start_ix + x2_ix + np.int64(x_len) * y2_ix]

    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    # points = sorted(points)               # order points by x, then by y
    # (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    #    #   if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
    #     raise ValueError('points do not form a rectangle')
    # if not x1 <= x <= x2 or not y1 <= y <= y2:
    #     raise ValueError('(x, y) not within the rectangle')

    if x == x2 or x == x1 or x1 == x2:
        if y == y1 or y == y2 or y1 == y2:
            return q11
        else:
            z = (q11 * (y2 - y) + q12 * (y - y1)) / (y2 - y1)
    elif y == y1 or y == y2 or y1 == y2:
        if x == x1 or x == x2 or x1 == x2:
            return q11
        else:
            z = (q11 * (x2 - x) + q21 * (x - x1)) / (x2 - x1)
    else:
        z = (q11 * (x2 - x) * (y2 - y) +
             q21 * (x - x1) * (y2 - y) +
             q12 * (x2 - x) * (y - y1) +
             q22 * (x - x1) * (y - y1)
             ) / ((x2 - x1) * (y2 - y1))
    return z
