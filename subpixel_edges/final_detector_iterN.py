import numpy as np

from subpixel_edges.edgepixel import EdgePixel
from subpixel_edges.edges_iterN import h_edges, v_edges


def main_iterN(F, threshold, iters, order):
    ep = EdgePixel()
    rows, cols = np.shape(F)

    RI = np.copy(F)

    for iterN in range(iters):
        # smooth image
        [x, y] = np.meshgrid(np.arange(cols), np.arange(rows))
        w = 0.75
        G = np.copy(RI)
        G[1:rows - 1, 1:cols - 1] = (RI[0:rows - 2, 0:cols - 2] + RI[0:rows - 2, 1:cols - 1] + RI[0:rows - 2, 2:cols] +
                                     RI[1:rows - 1, 0:cols - 2] + RI[1:rows - 1, 1:cols - 1] + RI[1:rows - 1, 2:cols] +
                                     RI[2:rows, 0:cols - 2] + RI[2:rows, 1:cols - 1] + RI[2:rows, 2:cols]) / 9

        # compute partial derivatives
        Gx = np.zeros((rows, cols))
        Gx[0: rows, 1: cols - 1] = 0.5 * (G[0: rows, 2: cols] - G[0: rows, 0: cols - 2])
        Gy = np.zeros((rows, cols))
        Gy[1: rows - 1, 0: cols] = 0.5 * (G[2: rows, 0: cols] - G[0: rows - 2, 0: cols])
        grad = np.sqrt(Gx ** 2 + Gy ** 2)

        # detect edge pixels with maximum Gy (not including margins)
        absGyInner = np.abs(Gy[5:rows - 5, 2: cols - 2])
        absGxInner = np.abs(Gx[2:rows - 2, 5: cols - 5])

        Ey = np.zeros((rows, cols), dtype=np.bool)
        Ex = np.zeros((rows, cols), dtype=np.bool)

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

        x_y, y_y, edges_y, nx_y, ny_y, i0_y, i1_y, curv_y, I, C, G = h_edges(RI, G, rows, Gx, Gy, w, edges_y, order,
                                                                             threshold, x, y, cols)
        x_x, y_x, edges_x, nx_x, ny_x, i0_x, i1_x, curv_x, RI, C, G = v_edges(RI, G, rows, Gx, Gy, w, edges_x, order,
                                                                              threshold, x, y, I, C)

        # compute final subimage
        RI[C > 0] = RI[C > 0] / C[C > 0]
        RI[C == 0] = G[C == 0]

    # save results
    ep.ny = np.concatenate((ny_y, ny_x), axis=0)
    ep.nx = np.concatenate((nx_y, nx_x), axis=0)

    ep.y = np.concatenate((y_y, y_x), axis=0)
    ep.x = np.concatenate((x_y, x_x), axis=0)

    ep.position = np.concatenate((edges_y, edges_x), axis=0)
    ep.curv = np.concatenate((curv_y, curv_x), axis=0)
    ep.i0 = np.concatenate((i0_y, i0_x), axis=0)
    ep.i1 = np.concatenate((i1_y, i1_x), axis=0)

    return ep
