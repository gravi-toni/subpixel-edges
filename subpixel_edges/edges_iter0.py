import numpy as np

from numba import njit


@njit(cache=True)
def h_edges(F, rows, Fx, Fy, edges, order):
    n_edges = np.shape(edges)[0]

    A = np.zeros((n_edges, 1))
    B = np.zeros((n_edges, 1))
    a = np.zeros((n_edges, 1))
    b = np.zeros((n_edges, 1))
    c = np.zeros((n_edges, 1))

    for k in range(n_edges):
        edge = edges[k]

        m1 = -1
        m2 = 1
        if Fx[edge] * Fy[edge] >= 0:
            m = 1
            l1 = 0
            r2 = 0
            min_l1 = -3
            max_r2 = 3
            l2 = 1
            r1 = -1
            max_l2 = 4
            min_r1 = -4
        else:
            m = -1
            l1 = -1
            r2 = 1
            min_l1 = -4
            max_r2 = 4
            l2 = 0
            r1 = 0
            max_l2 = 3
            min_r1 = -3

        if np.abs(Fx[edge]) < 1:
            l1 = -1
            l2 = 1
            r1 = -1
            r2 = 1

        while l1 > min_l1 and np.abs(Fy[edge - rows + l1]) >= np.abs(Fy[edge - rows + l1 - 1]):
            l1 -= 1

        while l2 < max_l2 and np.abs(Fy[edge - rows + l2]) >= np.abs(Fy[edge - rows + l2 + 1]):
            l2 += 1

        while m1 > -4 and np.abs(Fy[edge + m1]) >= np.abs(Fy[edge + m1 - 1]):
            m1 -= 1

        while m2 < 4 and np.abs(Fy[edge + m2]) >= np.abs(Fy[edge + m2 + 1]):
            m2 += 1

        while r1 > min_r1 and np.abs(Fy[edge + rows + r1]) >= np.abs(Fy[edge + rows + r1 - 1]):
            r1 -= 1

        while r2 < max_r2 and np.abs(Fy[edge + rows + r2]) >= np.abs(Fy[edge + rows + r2 + 1]):
            r2 += 1

        if m > 0:
            AA = (F[edge + m2] + F[edge + rows + r2]) / 2
            BB = (F[edge - rows + l1] + F[edge + m1]) / 2
        else:
            AA = (F[edge - rows + l2] + F[edge + m2]) / 2
            BB = (F[edge + m1] + F[edge + rows + r1]) / 2

        SL = 0
        SM = 0
        SR = 0

        for n in range(l1, l2 + 1):
            SL += F[edge - rows + n]

        for n in range(m1, m2 + 1):
            SM += F[edge + n]

        for n in range(r1, r2 + 1):
            SR += F[edge + rows + n]

        den = 2 * (AA - BB)

        if order == 2:
            if den != 0:
                c[k] = (SL + SR - 2 * SM + AA * (2 * m2 - l2 - r2) - BB * (2 * m1 - l1 - r1)) / den

        else:
            c[k] = 0

        if den != 0:
            b[k] = (SR - SL + AA * (l2 - r2) - BB * (l1 - r1)) / den
            a[k] = (2 * SM - AA * (1 + 2 * m2) - BB * (1 - 2 * m1)) / den - c[k] / 12

        A[k] = AA
        B[k] = BB

    return A, B, a, b, c


@njit(cache=True)
def v_edges(F, rows, Fx, Fy, edges, order):
    n_edges = np.shape(edges)[0]

    A = np.zeros((n_edges, 1))
    B = np.zeros((n_edges, 1))
    a = np.zeros((n_edges, 1))
    b = np.zeros((n_edges, 1))
    c = np.zeros((n_edges, 1))

    for k in range(n_edges):
        edge = edges[k]

        m1 = -1
        m2 = 1
        if Fx[edge] * Fy[edge] >= 0:
            m = 1
            l1 = 0
            r2 = 0
            min_l1 = -3
            max_r2 = 3
            l2 = 1
            r1 = -1
            max_l2 = 4
            min_r1 = -4
        else:
            m = -1
            l1 = -1
            r2 = 1
            min_l1 = -4
            max_r2 = 4
            l2 = 0
            r1 = 0
            max_l2 = 3
            min_r1 = -3

        if np.abs(Fy[edge]) < 1:
            l1 = -1
            l2 = 1
            r1 = -1
            r2 = 1

        while l1 > min_l1 and np.abs(Fx[edge - 1 + l1 * rows]) >= np.abs(Fx[edge - 1 + (l1 - 1) * rows]):
            l1 -= 1

        while l2 < max_l2 and np.abs(Fx[edge - 1 + l2 * rows]) >= np.abs(Fx[edge - 1 + (l2 + 1) * rows]):
            l2 += 1

        while m1 > -4 and np.abs(Fx[edge + m1 * rows]) >= np.abs(Fx[edge + (m1 - 1) * rows]):
            m1 -= 1

        while m2 < 4 and np.abs(Fx[edge + m2 * rows]) >= np.abs(Fx[edge + (m2 + 1) * rows]):
            m2 += 1

        while r1 > min_r1 and np.abs(Fx[edge + 1 + r1 * rows]) >= np.abs(Fx[edge + 1 + (r1 - 1) * rows]):
            r1 -= 1

        while r2 < max_r2 and np.abs(Fx[edge + 1 + r2 * rows]) >= np.abs(Fx[edge + 1 + (r2 + 1) * rows]):
            r2 += 1

        if m > 0:
            AA = (F[edge + m2 * rows] + F[edge + 1 + r2 * rows]) / 2
            BB = (F[edge - 1 + l1 * rows] + F[edge + m1 * rows]) / 2
        else:
            AA = (F[edge - 1 + l2 * rows] + F[edge + m2 * rows]) / 2
            BB = (F[edge + m1 * rows] + F[edge + 1 + r1 * rows]) / 2

        SL = 0
        SM = 0
        SR = 0

        for n in range(l1, l2 + 1):
            SL += F[edge - 1 + n * rows]

        for n in range(m1, m2 + 1):
            SM += F[edge + n * rows]

        for n in range(r1, r2 + 1):
            SR += F[edge + 1 + n * rows]

        # compute edge features
        den = 2 * (AA - BB)

        if order == 2:
            if den != 0:
                c[k] = (SL + SR - 2 * SM + AA * (2 * m2 - l2 - r2) - BB * (2 * m1 - l1 - r1)) / den

        else:
            c[k] = 0

        if den != 0:
            b[k] = (SR - SL + AA * (l2 - r2) - BB * (l1 - r1)) / den
            a[k] = (2 * SM - AA * (1 + 2 * m2) - BB * (1 - 2 * m1)) / den - c[k] / 12

        A[k] = AA
        B[k] = BB

    return A, B, a, b, c
