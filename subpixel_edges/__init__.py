import numpy as np

from subpixel_edges.final_detector_iter0 import main_iter0
from subpixel_edges.final_detector_iter1 import main_iter1
from subpixel_edges.final_detector_iterN import main_iterN


def init():
    img = np.zeros((8, 8))

    subpixel_edges(img, 15, 0, 2)
    subpixel_edges(img, 15, 1, 2)
    subpixel_edges(img, 15, 2, 2)


def subpixel_edges(img, threshold, iters, order):
    if iters == 0:
        return main_iter0(img, threshold, iters, order)
    elif iters == 1:
        return main_iter1(img, threshold, iters, order)
    elif iters > 1:
        return main_iterN(img, threshold, iters, order)
