import numpy as np
from numba import njit


@njit(cache=True)
def circle_grid(x, y, radius2, x_center, y_center, dx, dy):
    num_pixels = x.shape[0]

    if num_pixels > 0:
        p = np.zeros((num_pixels, 1))
        for n in range(0, num_pixels):
            grid = ((x[n] + dx - x_center) ** 2 + (y[n] + dy - y_center) ** 2 < radius2) * 1
            p[n] = np.mean(grid)

    else:
        p = np.zeros((0, 0))

    return p


@njit(cache=True)
def circle_horizontal_window(x_window_center, y_window_center, x_center, y_center, radius, inner_intensity,
                             outer_intensity, grid_resolution):
    # compute pixels completely outside or inside
    r2 = radius ** 2

    x = np.zeros(((np.arange(x_window_center - 4, x_window_center + 4 + 1)).size,
                  (np.arange(y_window_center - 1, y_window_center + 1 + 1)).size),
                 dtype=np.float64).T
    y = x.copy()
    x[:, :] = np.arange(x_window_center - 4, x_window_center + 4 + 1)
    y_vect = np.arange(y_window_center - 1, y_window_center + 1 + 1)
    y_vect = np.reshape(y_vect, (y_vect.shape[0], 1))
    y[:, :] = y_vect

    c = ((x - 0.5 - x_center) ** 2 + (y - 0.5 - y_center) ** 2 < r2).astype(np.float64)
    c = c + ((x - 0.5 - x_center) ** 2 + (y + 0.5 - y_center) ** 2 < r2)
    c = c + ((x + 0.5 - x_center) ** 2 + (y - 0.5 - y_center) ** 2 < r2)
    c = c + ((x + 0.5 - x_center) ** 2 + (y + 0.5 - y_center) ** 2 < r2)
    i = np.copy(c)

    bool_c0 = np.where(c.ravel() == 0)
    bool_c4 = np.where(c.ravel() == 4)
    i.ravel()[bool_c0] = outer_intensity
    i.ravel()[bool_c4] = inner_intensity

    # compute contour pixels
    delta = 1 / (grid_resolution - 1)

    dx = np.zeros(((np.arange(-0.5, 0.5, delta)).size, (np.arange(-0.5, 0.5, delta)).size), dtype=np.float32).T
    dy = dx.copy()
    dx[:, :] = np.arange(-0.5, 0.5, delta)
    dy_vect = np.arange(-0.5, 0.5, delta)
    dy_vect = np.reshape(dy_vect, (dy_vect.shape[0], 1))
    dy[:, :] = dy_vect

    bool_c04 = np.logical_and(c > 0, c < 4)

    i.ravel()[bool_c04.ravel()] = (outer_intensity +
                                   (inner_intensity - outer_intensity) * circle_grid(
                x.ravel()[(bool_c04.ravel())],
                y.ravel()[(bool_c04.ravel())], r2, x_center, y_center, dx, dy)).reshape((-1,))

    return i


@njit(cache=True)
def circle_vertical_window(x_window_center, y_window_center, x_center, y_center, radius, inner_intensity,
                           outer_intensity, grid_resolution):
    # compute pixels completely outside or inside
    r2 = radius ** 2

    x = np.zeros(((np.arange(x_window_center - 1, x_window_center + 1 + 1)).size,
                  (np.arange(y_window_center - 4, y_window_center + 4 + 1)).size),
                 dtype=np.float64).T
    y = x.copy()
    x[:, :] = np.arange(x_window_center - 1, x_window_center + 1 + 1)
    y_vect = np.arange((y_window_center - 4), (y_window_center + 4 + 1))
    y_vect = np.reshape(y_vect, (y_vect.shape[0], 1))
    y[:, :] = y_vect

    c = ((x - 0.5 - x_center) ** 2 + (y - 0.5 - y_center) ** 2 < r2).astype(np.float64)
    c = c + ((x - 0.5 - x_center) ** 2 + (y + 0.5 - y_center) ** 2 < r2)
    c = c + ((x + 0.5 - x_center) ** 2 + (y - 0.5 - y_center) ** 2 < r2)
    c = c + ((x + 0.5 - x_center) ** 2 + (y + 0.5 - y_center) ** 2 < r2)
    i = np.copy(c)

    bool_c0 = np.where(c.ravel() == 0)
    bool_c4 = np.where(c.ravel() == 4)
    i.ravel()[bool_c0] = outer_intensity
    i.ravel()[bool_c4] = inner_intensity

    # compute contour pixels
    delta = 1 / (grid_resolution - 1)

    dx = np.zeros(((np.arange(-0.5, 0.5, delta)).size, (np.arange(-0.5, 0.5, delta)).size), dtype=np.float32).T
    dy = dx.copy()
    dx[:, :] = np.arange(-0.5, 0.5, delta)
    dy_vect = np.arange(-0.5, 0.5, delta)
    dy_vect = np.reshape(dy_vect, (dy_vect.shape[0], 1))
    dy[:, :] = dy_vect

    bool_c04 = np.logical_and(c > 0, c < 4)
    grid = circle_grid(
                x.ravel()[(bool_c04.ravel())],
                y.ravel()[(bool_c04.ravel())], r2, x_center, y_center, dx, dy)

    i.ravel()[bool_c04.ravel()] = (outer_intensity + (inner_intensity - outer_intensity) * grid).reshape((-1,))

    return i
