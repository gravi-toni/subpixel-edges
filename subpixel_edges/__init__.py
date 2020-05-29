import numpy as np

from subpixel_edges.final_detector_iter0 import main_iter0
from subpixel_edges.final_detector_iter1 import main_iter1
from subpixel_edges.final_detector_iterN import main_iterN


def init():
    """
    Run all the methods to trigger JIT compilation. Only needed the first time
    the code is executed. This method may be removed in future releases, for example if
    AOT compilation is used instead.
    """
    img = np.zeros((8, 8))

    subpixel_edges(img, 15, 0, 2)
    subpixel_edges(img, 15, 1, 2)
    subpixel_edges(img, 15, 2, 2)


def subpixel_edges(img, threshold, iters, order):
    """
    Detects subpixel features for each pixel belonging to an edge in `img`.

    The subpixel edge detection used the method published in the following paper:
    "Accurate Subpixel Edge Location Based on Partial Area Effect"
    http://www.sciencedirect.com/science/article/pii/S0262885612001850

    Parameters
    ----------
    img: ndarray
        A grayscale image.
    threshold: int or float
        Specifies the minimum difference of intensity at both
        sides of a pixel to be considered as an edge.
    iters: int
        Specifies how many smoothing iterations are needed
        to find the final edges:
            0:  Oriented to noise free images. No previous smoothing on
                the image. The detection is applied on the original
                image values (section 3 of the paper).
            1:  Oriented to low-noise images. The detection is applied
                on the image previously smoothed by a 3x3 mask
                (default) (sections 4 and 5 of the paper)
            >1: Oriented to high-noise images. Several stages of
                smoothing + detection + synthetic image creation are
                applied (section 6 of the paper). A few iterations are
                normally enough.
    order: int
        Specifies the order of the edges to find:
            1:  first order edges (straight lines)
            2:  second order edges (default)

    Returns
    -------
    An instance of EdgePixel
    """
    if iters == 0:
        return main_iter0(img, threshold, iters, order)
    elif iters == 1:
        return main_iter1(img, threshold, iters, order)
    elif iters > 1:
        for iterN in range(iters):
            ep, img = main_iterN(img, threshold, iters, order)
        return ep
