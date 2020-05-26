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
