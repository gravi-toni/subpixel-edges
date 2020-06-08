import os

import cv2
import numpy as np
import pytest

from subpixel_edges import subpixel_edges
from subpixel_edges.edgepixel import EdgePixel


@pytest.mark.parametrize('iters', [0, 1, 2])
class TestIters:
    """
    Tests that the results from the Python implementation do not change.
    """
    this_path = os.path.dirname(os.path.realpath(__file__))

    def read_image(self, filename):
        img = cv2.imread(os.path.join(self.this_path, filename))
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)

    def read_edges(self, filename):
        return EdgePixel.load(os.path.join(self.this_path, filename))

    def test_iters(self, iters):
        test_edges = self.read_edges(f'data/lena_{iters}.npz')

        img_gray = self.read_image('images/lena.png')
        edges = subpixel_edges(img_gray, 25, iters, 2)

        assert np.allclose(edges.position, test_edges.position)
        assert np.allclose(edges.x, test_edges.x)
        assert np.allclose(edges.y, test_edges.y)
        assert np.allclose(edges.nx, test_edges.nx)
        assert np.allclose(edges.ny, test_edges.ny)
        assert np.allclose(edges.curv, test_edges.curv)
        assert np.allclose(edges.i0, test_edges.i0)
        assert np.allclose(edges.i1, test_edges.i1)
