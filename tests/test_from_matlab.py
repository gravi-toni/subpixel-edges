import os

import cv2
import numpy as np

from scipy import io

from subpixel_edges import subpixel_edges
from subpixel_edges.edgepixel import EdgePixel


class TestFromMatlab:
    """
    Tests that the results from the Python implementation are reasonably
    compatible with the results from the reference MATLAB implementation.

    The image MATLAB file is created from Matlab with:

    > image = imread('images/ring.tif');
    > save('data/ring.mat', 'image', '-v7');

    The edges MATLAB files are created from Matlab with:

    > res0 = struct(subpixelEdges(image, 25, 'Order', 2, 'SmoothingIter', 0));
    > save('data/ring_0.mat', '-struct', 'res0', '-v7');
    """
    this_path = os.path.dirname(os.path.realpath(__file__))

    def read_image(self, filename):
        img = cv2.imread(os.path.join(self.this_path, filename))
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)

    def read_edges(self, filename):
        return EdgePixel.load(os.path.join(self.this_path, filename), format='mat')

    def test_image_read_same(self):
        img_gray = self.read_image('images/ring.tif')

        img_from_mat = io.loadmat(os.path.join(self.this_path, 'data/ring.mat'))['image']

        assert img_from_mat.shape == img_gray.shape
        assert np.array_equiv(img_from_mat, img_gray)

    def test_iter0(self):
        """
        When `iters` = 0, the results are perfectly consistent.
        """
        test_edges = self.read_edges('data/ring_0.mat')

        img_gray = self.read_image('images/ring.tif')
        edges = subpixel_edges(img_gray, 25, 0, 2)

        assert np.array_equiv(edges.position, test_edges.position.ravel('F') - 1)
        assert np.allclose(edges.x, test_edges.x.ravel('F') - 1)
        assert np.allclose(edges.y, test_edges.y.ravel('F') - 1)
        assert np.allclose(edges.nx, test_edges.nx.ravel('F'))
        assert np.allclose(edges.ny, test_edges.ny.ravel('F'))
        assert np.allclose(edges.curv, test_edges.curv.ravel('F'))
        assert np.allclose(edges.i0, test_edges.i0.ravel('F'))
        assert np.allclose(edges.i1, test_edges.i1.ravel('F'))

    def test_iter1(self):
        """
        When `iters` = 1, the results are still consistent except for one position value.
        """
        test_edges = self.read_edges('data/ring_1.mat')

        img_gray = self.read_image('images/ring.tif')
        edges = subpixel_edges(img_gray, 25, 1, 2)

        mask = np.ones(len(edges.position), dtype=bool)
        # Excluded values that are known to be different
        mask[[258]] = False

        assert np.array_equiv(edges.position[mask], test_edges.position[mask].ravel('F') - 1)
        assert np.allclose(edges.x[mask], test_edges.x[mask].ravel('F') - 1)
        assert np.allclose(edges.y[mask], test_edges.y[mask].ravel('F') - 1)
        assert np.allclose(edges.nx[mask], test_edges.nx[mask].ravel('F'))
        assert np.allclose(edges.ny[mask], test_edges.ny[mask].ravel('F'))
        assert np.allclose(edges.curv[mask], test_edges.curv[mask].ravel('F'))
        assert np.allclose(edges.i0[mask], test_edges.i0[mask].ravel('F'))
        assert np.allclose(edges.i1[mask], test_edges.i1[mask].ravel('F'))

    def test_iter2(self):
        """
        When `iters` = 2, numerical effects start to be significant.
        """
        test_edges = self.read_edges('data/ring_2.mat')

        img_gray = self.read_image('images/ring.tif')
        edges = subpixel_edges(img_gray, 25, 2, 2)

        mask = np.ones(len(edges.position), dtype=bool)
        # Excluded values that are known to be different
        mask[[257, 258]] = False

        assert np.array_equiv(edges.position[mask], test_edges.position[mask].ravel('F') - 1)
        assert np.allclose(edges.x[mask], test_edges.x[mask].ravel('F') - 1,
                           atol=1e-2, rtol=1e-5)
        assert np.allclose(edges.y[mask], test_edges.y[mask].ravel('F') - 1,
                           atol=1e-1, rtol=1e-4)
        assert np.allclose(edges.nx[mask], test_edges.nx[mask].ravel('F'),
                           atol=1e-2, rtol=1e-5)
        assert np.allclose(edges.ny[mask], test_edges.ny[mask].ravel('F'),
                           atol=1e-2, rtol=1e-5)
        assert np.allclose(edges.curv[mask], test_edges.curv[mask].ravel('F'),
                           atol=1e-1, rtol=1e-4)
        assert np.allclose(edges.i0[mask], test_edges.i0[mask].ravel('F'),
                           atol=1e-1, rtol=1e-4)
        assert np.allclose(edges.i1[mask], test_edges.i1[mask].ravel('F'),
                           atol=1e-1, rtol=1e-4)
