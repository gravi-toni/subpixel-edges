# subpixel-edges

A pure Python implementation of the subpixel edge location algorithm from https://doi.org/10.1016/j.imavis.2012.10.005

The reference implementation can be found on from https://it.mathworks.com/matlabcentral/fileexchange/48908-accurate-subpixel-edge-location


# Installation

`pip install subpixel-edges`

# Quick start

For a quick overview of the code functionalities, install the following packages first:

1) pip install subpixel-edges
2) pip install opencv-python
3) pip install matplotlib

Then go into the directory you want to use and copy the image you want to analyze (let's say 'lena.png'). 
Now open a python console and execute the following commands:

1) import cv2
2) import matplotlib.pyplot as plt
3) from subpixel_edges import subpixel_edges
4) ((optional)) help(subpixel_edges) 
5) img = cv2.imread("lena.png")
6) img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
7) edges = subpixel_edges(img_gray,25,0,2)
8) plt.imshow(img)
9) plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
10)plt.show()

## Development

```
git clone https://github.com/gravi-toni/subpixel-edges.git
pip install -e .
```

To run the tests (includes OpenCV):

`pip install -e .[tests]`

To run the examples (includes OpenCV):

`pip install -e .[examples]`
