import numpy as np

from numba import njit


@njit(cache=True)
def h_edges(x, y, F, G, rows, Gx, Gy, w, maxValidOffset, edges, order, threshold, FF, GG):
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
        if Gx[edge] * Gy[edge] >= 0:
            m = 1
            l1 = -1
            r2 = 1
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
            l2 = 1
            r1 = -1
            max_l2 = 3
            min_r1 = -3

        while l1 > min_l1 and np.abs(Gy[edge - rows + l1]) >= np.abs(Gy[edge - rows + l1 - 1]):
            l1 -= 1

        while l2 < max_l2 and np.abs(Gy[edge - rows + l2]) >= np.abs(Gy[edge - rows + l2 + 1]):
            l2 += 1

        while m1 > -4 and np.abs(Gy[edge + m1]) >= np.abs(Gy[edge + m1 - 1]):
            m1 -= 1

        while m2 < 4 and np.abs(Gy[edge + m2]) >= np.abs(Gy[edge + m2 + 1]):
            m2 += 1

        while r1 > min_r1 and np.abs(Gy[edge + rows + r1]) >= np.abs(Gy[edge + rows + r1 - 1]):
            r1 -= 1

        while r2 < max_r2 and np.abs(Gy[edge + rows + r2]) >= np.abs(Gy[edge + rows + r2 + 1]):
            r2 += 1

        if m > 0:
            AA = (GG[edge + m2] + GG[edge + rows + r2]) / 2
            BB = (GG[edge - rows + l1] + GG[edge + m1]) / 2
        else:
            AA = (GG[edge - rows + l2] + GG[edge + m2]) / 2
            BB = (GG[edge + m1] + GG[edge + rows + r1]) / 2

        # search for a second close edge
        u_border = False
        d_border = False

        if m1 > -4:
            partial = abs(Gy[edge + m1 - 2])
            if partial > abs(Gy[edge] / 4) and partial > threshold / 2:
                u_border = True

        if m2 < 4:
            partial = abs(Gy[edge + m2 + 2])
            if partial > abs(Gy[edge] / 4) and partial > threshold / 2:
                d_border = True

        SL = 0
        SM = 0
        SR = 0

        if u_border or d_border:
            jj = np.int_(np.floor((edges[k] - 1) / rows) + 1)
            i = np.int_(edges[k] - rows * (jj - 1))
            rimvt = np.copy(F[i - 5:i + 6, jj - 3:jj + 2])

            if u_border:
                if m > 0:
                    BB = (FF[edge + m1] + FF[edge - rows + l1]) / 2
                    p = 1
                else:
                    BB = (FF[edge + m1] + FF[edge + rows + r1]) / 2
                    p = 0

                if Gy[edge - 2 * rows + l1 + p] * Gy[edge] > 0:
                    ll = l1 + p - 1
                else:
                    ll = l1 + p

                if Gy[edge + 2 * rows + r1 + 1 - p] * Gy[edge] > 0:
                    rr = r1 - p
                else:
                    rr = r1 + 1 - p

                rimvt[0:ll + 6, 0] = BB
                rimvt[0:l1 + 6, 1] = BB
                rimvt[0:m1 + 6, 2] = BB
                rimvt[0:r1 + 6, 3] = BB
                rimvt[0:rr + 6, 4] = BB

                l1 = -3 + m
                m1 = -3
                r1 = -3 - m

            if d_border:
                if m > 0:
                    AA = (FF[edge + m2] + FF[edge + rows + r2]) / 2
                    p = 1
                else:
                    AA = (FF[edge + m2] + FF[edge - rows + l2]) / 2
                    p = 0

                if Gy[edge - 2 * rows + l2 + p - 1] * Gy[edge] > 0:
                    ll = l2 + p
                else:
                    ll = l2 + p - 1

                if Gy[edge + 2 * rows + r2 - p] * Gy[edge] > 0:
                    rr = r2 + 1 - p
                else:
                    rr = r2 - p

                rimvt[ll + 5:11, 0] = AA
                rimvt[l2 + 5:11, 1] = AA
                rimvt[m2 + 5:11, 2] = AA
                rimvt[r2 + 5:11, 3] = AA
                rimvt[rr + 5:11, 4] = AA

                l2 = 3 + m
                m2 = 3
                r2 = 3 - m

            rimv2 = (rimvt[0:9, 0:3] + rimvt[0:9, 1:4] + rimvt[0:9, 2:5] +
                     rimvt[1:10, 0:3] + rimvt[1:10, 1:4] + rimvt[1:10, 2:5] +
                     rimvt[2:11, 0:3] + rimvt[2:11, 1:4] + rimvt[2:11, 2:5]) / 9

            for n in range(l1 + 5 - 1, l2 + 5):
                SL = SL + rimv2[n, 0]
            for n in range(m1 + 5 - 1, m2 + 5):
                SM = SM + rimv2[n, 1]
            for n in range(r1 + 5 - 1, r2 + 5):
                SR = SR + rimv2[n, 2]
        else:
            for n in range(l1, l2 + 1):
                SL = SL + GG[edge - rows + n]
            for n in range(m1, m2 + 1):
                SM = SM + GG[edge + n]
            for n in range(r1, r2 + 1):
                SR = SR + GG[edge + rows + n]

        # compute edge features
        den = 2 * (AA - BB)
        if order == 2:
            if den != 0:
                c[k] = np.divide(SL + SR - 2 * SM + AA * (2 * m2 - l2 - r2) - BB * (2 * m1 - l1 - r1), den)
        else:
            c[k] = 0

        if den != 0:
            b[k] = np.divide(SR - SL + AA * (l2 - r2) - BB * (l1 - r1), den)
            a[k] = np.divide(2 * SM - AA * (1 + 2 * m2) - BB * (1 - 2 * m1), den) - w * c[k]
            A[k] = AA
            B[k] = BB

    # remove invalid values
    valid = np.abs(a) < maxValidOffset
    valid = valid.reshape((-1,))
    edges = edges[valid]

    if len(edges) != 0:
        A = A[valid]
        B = B[valid]
        a = a[valid]
        b = b[valid]
        c = c[valid]

    nn = np.ones((np.shape(edges)[0], 1))
    nn[Gy[edges] < 0] = -1

    x = x[edges]
    y = y[edges] - a.transpose().ravel()

    b = b.reshape((b.shape[0]), 1)
    nx = np.sign(A - B) / np.sqrt(1 + b ** 2) * b
    ny = np.sign(A - B) / np.sqrt(1 + b ** 2)

    curv = 2 * c * nn / ((1 + b ** 2) ** 1.5)

    i0 = np.minimum(A, B)
    i1 = np.maximum(A, B)

    return edges, x, y, nx.transpose().ravel(), ny.transpose().ravel(), curv.reshape(
        (-1,)), i0.transpose().ravel(), i1.transpose().ravel()


@njit(cache=True)
def v_edges(x, y, F, G, rows, Gx, Gy, w, maxValidOffset, edges, order, threshold, FF, GG):
    n_edges = np.shape(edges)[0]

    A = np.zeros((n_edges, 1))
    B = np.zeros((n_edges, 1))
    a = np.zeros((n_edges, 1))
    b = np.zeros((n_edges, 1))
    c = np.zeros((n_edges, 1))

    # compute all vertical edges
    for k in range(n_edges):
        edge = edges[k]

        # compute window floating limits
        m1 = -1
        m2 = 1

        if Gx[edge] * Gy[edge] >= 0:
            m = 1
            l1 = -1
            r2 = 1
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
            l2 = 1
            r1 = -1
            max_l2 = 3
            min_r1 = -3

        while l1 > min_l1 and np.abs(Gx[edge - 1 + l1 * rows]) >= np.abs(Gx[edge - 1 + (l1 - 1) * rows]):
            l1 = l1 - 1

        while l2 < max_l2 and np.abs(Gx[edge - 1 + l2 * rows]) >= np.abs(Gx[edge - 1 + (l2 + 1) * rows]):
            l2 = l2 + 1

        while m1 > -4 and np.abs(Gx[edge + m1 * rows]) >= np.abs(Gx[edge + (m1 - 1) * rows]):
            m1 = m1 - 1

        while m2 < 4 and np.abs(Gx[edge + m2 * rows]) >= np.abs(Gx[edge + (m2 + 1) * rows]):
            m2 = m2 + 1

        while r1 > min_r1 and np.abs(Gx[edge + 1 + r1 * rows]) >= np.abs(Gx[edge + 1 + (r1 - 1) * rows]):
            r1 = r1 - 1

        while r2 < max_r2 and np.abs(Gx[edge + 1 + r2 * rows]) >= np.abs(Gx[edge + 1 + (r2 + 1) * rows]):
            r2 = r2 + 1

        # compute intensities
        if m > 0:
            AA = (GG[edge + m2 * rows] + GG[edge + 1 + r2 * rows]) / 2
            BB = (GG[edge - 1 + l1 * rows] + GG[edge + m1 * rows]) / 2
        else:
            AA = (GG[edge - 1 + l2 * rows] + GG[edge + m2 * rows]) / 2
            BB = (GG[edge + m1 * rows] + GG[edge + 1 + r1 * rows]) / 2

        # search for a second close edge
        u_border = False
        d_border = False

        if m1 > -4:
            partial = np.abs(Gx[edge + (m1 - 2) * rows])
            if partial > np.abs(Gx[edge] / 4) and partial > threshold / 2:
                u_border = True

        if m2 < 4:
            partial = np.abs(Gx[edge + (m2 + 2) * rows])
            if partial > np.abs(Gx[edge] / 4) and partial > threshold / 2:
                d_border = True

        SL = 0
        SM = 0
        SR = 0

        if u_border or d_border:
            j = np.int_(np.floor((edges[k] - 1) / rows) + 1)
            i = np.int_(edges[k] - rows * (j - 1))
            rimvt = np.copy(F[i - 2:i + 3, j - 6:j + 5])
            if u_border:
                if m > 0:
                    BB = (FF[edge + m1 * rows] + FF[edge - 1 + l1 * rows]) / 2
                    p = 1
                else:
                    BB = (FF[edge + m1 * rows] + FF[edge + 1 + r1 * rows]) / 2
                    p = 0

                if Gx[edge - 2 + (l1 + p) * rows] * Gx[edge] > 0:
                    ll = l1 + p - 1
                else:
                    ll = l1 + p

                if Gx[edge + 2 + (r1 + 1 - p) * rows] * Gx[edge] > 0:
                    rr = r1 - p
                else:
                    rr = r1 + 1 - p

                rimvt[0, 0:ll + 6] = BB
                rimvt[1, 0:l1 + 6] = BB
                rimvt[2, 0:m1 + 6] = BB
                rimvt[3, 0:r1 + 6] = BB
                rimvt[4, 0:rr + 6] = BB

                l1 = -3 + m
                m1 = -3
                r1 = -3 - m

            if d_border:
                if m > 0:
                    AA = (FF[edge + m2 * rows] + FF[edge + 1 + r2 * rows]) / 2
                    p = 1
                else:
                    AA = (FF[edge + m2 * rows] + FF[edge - 1 + l2 * rows]) / 2
                    p = 0

                if Gx[edge - 2 + (l2 + p - 1) * rows] * Gx[edge] > 0:
                    ll = l2 + p
                else:
                    ll = l2 + p - 1

                if Gx[edge + 2 + (r2 - p) * rows] * Gx[edge] > 0:
                    rr = r2 + 1 - p
                else:
                    rr = r2 - p

                rimvt[0, ll + 6:11] = AA
                rimvt[1, l2 + 6:11] = AA
                rimvt[2, m2 + 6:11] = AA
                rimvt[3, r2 + 6:11] = AA
                rimvt[4, rr + 6:11] = AA

                l2 = 3 + m
                m2 = 3
                r2 = 3 - m

            rimv2 = (rimvt[0:3, 0:9] + rimvt[1:4, 0:9] + rimvt[2:5, 0:9] + \
                     rimvt[0:3, 1:10] + rimvt[1:4, 1:10] + rimvt[2:5, 1:10] + \
                     rimvt[0:3, 2:11] + rimvt[1:4, 2:11] + rimvt[2:5, 2:11]) / 9

            for n in range(l1 + 5 - 1, l2 + 5):
                SL = SL + rimv2[0, n]
            for n in range(m1 + 5 - 1, m2 + 5):
                SM = SM + rimv2[1, n]
            for n in range(r1 + 5 - 1, r2 + 5):
                SR = SR + rimv2[2, n]
        else:

            for n in range(l1, l2 + 1):
                SL = SL + GG[edge - 1 + n * rows]
            for n in range(m1, m2 + 1):
                SM = SM + GG[edge + n * rows]
            for n in range(r1, r2 + 1):
                SR = SR + GG[edge + 1 + n * rows]

        # compute edge features
        den = 2 * (AA - BB)
        if order == 2:
            if den != 0:
                c[k] = np.divide((SL + SR - 2 * SM + AA * (2 * m2 - l2 - r2) - BB * (2 * m1 - l1 - r1)), den)
        else:
            c[k] = 0

        if den != 0:
            b[k] = np.divide((SR - SL + AA * (l2 - r2) - BB * (l1 - r1)), den)
            a[k] = np.divide((2 * SM - AA * (1 + 2 * m2) - BB * (1 - 2 * m1)), den) - w * c[k]
            A[k] = AA
            B[k] = BB

    # remove invalid values
    valid = np.abs(a) < maxValidOffset
    valid = valid.reshape((-1,))
    edges = edges[valid]
    if len(edges) != 0:
        A = A[valid]
        B = B[valid]
        a = a[valid]
        b = b[valid]
        c = c[valid]

    x = x[edges] - a.transpose().ravel()
    y = y[edges]

    nn = np.ones((np.shape(edges)[0], 1))
    nn[Gx[edges] < 0] = -1

    b = b.reshape((b.shape[0]), 1)
    nx = np.sign(A - B) / np.sqrt(1 + b ** 2)
    ny = np.sign(A - B) / np.sqrt(1 + b ** 2) * b

    curv = 2 * c * nn / ((1 + b ** 2) ** 1.5)

    i0 = np.minimum(A, B)
    i1 = np.maximum(A, B)

    return edges, x.transpose().ravel(), y.transpose().ravel(), nx.transpose().ravel(), ny.transpose().ravel(), \
           curv.reshape((-1,)), i0.transpose().ravel(), i1.transpose().ravel()
