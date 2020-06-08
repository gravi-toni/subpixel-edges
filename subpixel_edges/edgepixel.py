import numpy as np
import scipy


class EdgePixel:
    """
    Class describing pixel features.

    position:
        1D-index inside image
    x, y:
        subpixel position
    nx, ny:
        normal vector (normalized)
    curv:
        curvature
    i0, i1:
        intensities at both sides
    """
    def __init__(self, position=None, x=None, y=None, nx=None, ny=None, curv=None, i0=None, i1=None):
        
        self.position = position  # 1D index inside image
        self.x = x                # X subpixel position
        self.y = y                # Y subpixel position
        self.nx = nx              # normal vector (normalized)
        self.ny = ny              # normal vector (normalized)
        self.curv = curv          # curvature
        self.i0 = i0              # intensities
        self.i1 = i1              # intensities

    def save(self, filename, format='npz'):
        if format == 'npz':
            np.savez(filename, **vars(self))
        else:
            raise ValueError('File format "%s" not supported' % format)

    @classmethod
    def load(cls, filename, format='npz'):
        if format == 'npz':
            edges = np.load(filename)

            return EdgePixel(
                position=edges['position'],
                x=edges['x'],
                y=edges['y'],
                nx=edges['nx'],
                ny=edges['ny'],
                curv=edges['curv'],
                i0=edges['i0'],
                i1=edges['i1']
            )
        elif format == 'mat':
            edges = scipy.io.loadmat(filename)

            return EdgePixel(
                position=edges['position'].ravel('F'),
                x=edges['x'].ravel('F'),
                y=edges['y'].ravel('F'),
                nx=edges['nx'].ravel('F'),
                ny=edges['ny'].ravel('F'),
                curv=edges['curv'].ravel('F'),
                i0=edges['i0'].ravel('F'),
                i1=edges['i1'].ravel('F')
            )
        else:
            raise ValueError('File format "%s" not supported' % format)
