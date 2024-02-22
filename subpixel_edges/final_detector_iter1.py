import numpy as np

from scipy import ndimage

from subpixel_edges.edgepixel import EdgePixel
from subpixel_edges.edges_iter1 import h_edges, v_edges


def main_iter1(F, threshold, iters, order):
    # smooth image
    H = np.ones((3, 3), np.float32) / 9
    w = (1 + 24 * H[1, 2] + 48 * H[1, 1]) / 12
    G = ndimage.convolve(F, H, mode='constant')

    # initialization
    rows, cols = np.shape(G)
    [x, y] = np.meshgrid(np.arange(cols), np.arange(rows))
    ep = EdgePixel()
    max_valid_offset = 1

    # compute partial derivatives
    Gx = np.zeros((rows, cols))
    Gx[0: rows, 1: cols - 1] = 0.5 * (G[0: rows, 2: cols] - G[0: rows, 0: cols - 2])
    Gy = np.zeros((rows, cols))
    Gy[1: rows - 1, 0: cols] = 0.5 * (G[2: rows, 0: cols] - G[0: rows - 2, 0: cols])
    grad = np.sqrt(Gx ** 2 + Gy ** 2)

    # detect edge pixels with maximum Gy (not including margins)
    absGyInner = np.abs(Gy[5:rows - 5, 2: cols - 2])
    absGxInner = np.abs(Gx[2:rows - 2, 5: cols - 5])

    Ey = np.zeros((rows, cols), dtype=np.bool_)
    Ex = np.zeros((rows, cols), dtype=np.bool_)

    Ey[5: rows - 5, 2: cols - 2] = np.logical_and.reduce([
        grad[5: rows - 5, 2: cols - 2] > threshold,
        absGyInner >= np.abs(Gx[5: rows - 5, 2: cols - 2]),
        absGyInner >= np.abs(Gy[4: rows - 6, 2: cols - 2]),
        absGyInner > np.abs(Gy[6: rows - 4, 2: cols - 2])
    ])

    Ex[2: rows - 2, 5: cols - 5] = np.logical_and.reduce([
        grad[2: rows - 2, 5: cols - 5] > threshold,
        absGxInner > np.abs(Gy[2: rows - 2, 5: cols - 5]),
        absGxInner >= np.abs(Gx[2: rows - 2, 4: cols - 6]),
        absGxInner > np.abs(Gx[2: rows - 2, 6: cols - 4])
    ])

    Ey = Ey.ravel('F')
    Ex = Ex.ravel('F')
    y = y.ravel('F')
    x = x.ravel('F')

    edges_y = (x[Ey] * rows + y[Ey])
    edges_x = (x[Ex] * rows + y[Ex])

    Gx = Gx.ravel('F')
    Gy = Gy.ravel('F')

    FF = F.ravel('F')
    GG = G.ravel('F')

    edges_y, x_y, y_y, nx_y, ny_y, curv_y, i0_y, i1_y = h_edges(x, y, F, G, rows, Gx, Gy, w, max_valid_offset, edges_y,
                                                                order, threshold, FF, GG)
    edges_x, x_x, y_x, nx_x, ny_x, curv_x, i0_x, i1_x = v_edges(x, y, F, G, rows, Gx, Gy, w, max_valid_offset, edges_x,
                                                                order, threshold, FF, GG)

    ep.ny = np.concatenate((ny_y, ny_x), axis=0)
    ep.nx = np.concatenate((nx_y, nx_x), axis=0)

    ep.y = np.concatenate((y_y, y_x), axis=0)
    ep.x = np.concatenate((x_y, x_x), axis=0)

    ep.position = np.concatenate((edges_y, edges_x), axis=0)
    ep.curv = np.concatenate((curv_y, curv_x), axis=0)
    ep.i0 = np.concatenate((i0_y, i0_x), axis=0)
    ep.i1 = np.concatenate((i1_y, i1_x), axis=0)

    return ep
