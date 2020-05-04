import os

import cv2
import matplotlib.pyplot as plt

from time import time

from subpixel_edges import init, subpixel_edges


if __name__ == '__main__':
    this_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(this_path, 'images/saturn.jpg'))
    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)

    iters = 1
    threshold = 15
    order = 2

    print("Initializing...")
    now = time()
    init()
    elapsed = time() - now
    print("Initialization time: ", elapsed)

    print("Processing...")
    now = time()
    edges = subpixel_edges(img_gray, threshold, iters, order)
    elapsed = time() - now
    print("Processing time: ", elapsed)

    plt.imshow(img[..., ::-1])
    # seg = 0.6
    plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, width=0.01, scale=80)
    # plt.plot(ep.x, ep.y, ".", markersize=1, color='r')
    plt.show()
