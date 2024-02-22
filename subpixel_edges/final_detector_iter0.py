import numpy as np

from subpixel_edges.edgepixel import EdgePixel
from subpixel_edges.edges_iter0 import h_edges, v_edges


def main_iter0(F, threshold, iters, order):
    ep = EdgePixel()

    rows, cols = np.shape(F)
    [x, y] = np.meshgrid(np.arange(cols), np.arange(rows))

    Fx = np.zeros((rows, cols))
    Fx[0: rows, 1: cols - 1] = 0.5 * (F[0: rows, 2: cols] - F[0: rows, 0: cols - 2])
    Fy = np.zeros((rows, cols))
    Fy[1: rows - 1, 0: cols] = 0.5 * (F[2: rows, 0: cols] - F[0: rows - 2, 0: cols])
    grad = np.sqrt(Fx ** 2 + Fy ** 2)

    abs_Fy_inner = np.abs(Fy[5:rows - 5, 2: cols - 2])
    abs_Fx_inner = np.abs(Fx[2:rows - 2, 5: cols - 5])

    Ey = np.zeros((rows, cols), dtype=np.bool_)
    Ex = np.zeros((rows, cols), dtype=np.bool_)

    Ey[5: rows - 5, 2: cols - 2] = np.logical_and.reduce([
        grad[5: rows - 5, 2: cols - 2] > threshold,
        abs_Fy_inner >= np.abs(Fx[5: rows - 5, 2: cols - 2]),
        abs_Fy_inner >= np.abs(Fy[4: rows - 6, 2: cols - 2]),
        abs_Fy_inner > np.abs(Fy[6: rows - 4, 2: cols - 2])
    ])

    Ex[2: rows - 2, 5: cols - 5] = np.logical_and.reduce([
        grad[2: rows - 2, 5: cols - 5] > threshold,
        abs_Fx_inner > np.abs(Fy[2: rows - 2, 5: cols - 5]),
        abs_Fx_inner >= np.abs(Fx[2: rows - 2, 4: cols - 6]),
        abs_Fx_inner > np.abs(Fx[2: rows - 2, 6: cols - 4])
    ])

    Ey = Ey.ravel('F')
    Ex = Ex.ravel('F')
    y = y.ravel('F')
    x = x.ravel('F')

    edges_y = (x[Ey] * rows + y[Ey])
    edges_x = (x[Ex] * rows + y[Ex])

    F = F.ravel('F')
    Fx = Fx.ravel('F')
    Fy = Fy.ravel('F')

    ny = np.ones((np.shape(edges_y)[0], 1))
    ny[Fy[edges_y] < 0] = -1
    nx = np.ones((np.shape(edges_x)[0], 1))
    nx[Fx[edges_x] < 0] = -1

    Ay, By, ay, by, cy = h_edges(F, rows, Fx, Fy, edges_y, order)
    Ax, Bx, ax, bx, cx = v_edges(F, rows, Fx, Fy, edges_x, order)

    ay = ay.ravel('F')
    ax = ax.ravel('F')

    ep.position = np.r_[edges_y, edges_x]
    ep.x = np.r_[x[edges_y], x[edges_x] - ax]
    ep.y = np.r_[y[edges_y] - ay, y[edges_x]]

    ep.nx = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by ** 2) * by,
        np.sign(Ax - Bx) / np.sqrt(1 + bx ** 2),
    ]
    ep.ny = np.r_[
        np.sign(Ay - By) / np.sqrt(1 + by ** 2),
        np.sign(Ax - Bx) / np.sqrt(1 + bx ** 2) * bx,
    ]
    ep.curv = np.r_[
        2 * cy * ny / ((1 + by ** 2) ** 1.5),
        2 * cx * nx / ((1 + bx ** 2) ** 1.5),
    ]
    ep.i0 = np.r_[np.minimum(Ay, By), np.minimum(Ax, Bx)]
    ep.i1 = np.r_[np.maximum(Ay, By), np.maximum(Ax, Bx)]

    # # erase elements outside the image size
    index_to_erase1 = np.union1d(np.where(ep.x > cols), np.where(ep.y > rows))
    index_to_erase2 = np.union1d(np.where(ep.x < 0), np.where(ep.y < 0))
    
    index_to_erase = np.union1d(index_to_erase1, index_to_erase2)
    
    ep.x = np.delete(ep.x, index_to_erase)
    ep.y = np.delete(ep.y, index_to_erase)
    ep.nx = np.delete(ep.nx, index_to_erase)
    ep.ny = np.delete(ep.ny, index_to_erase)
    ep.i0 = np.delete(ep.i0, index_to_erase)
    ep.i1 = np.delete(ep.i1, index_to_erase)
    ep.position = np.delete(ep.position, index_to_erase)
    ep.curv = np.delete(ep.curv, index_to_erase)

    return ep
