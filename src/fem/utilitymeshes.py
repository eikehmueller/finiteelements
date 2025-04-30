"""Computational meshes"""

import numpy as np
from fem.mesh import Mesh

__all__ = ["RectangleMesh", "TriangleMesh"]


class RectangleMesh(Mesh):
    """Triangular mesh ontained by refinement of a single rectangle"""

    def __init__(self, Lx=1.0, Ly=1.0, nref=0):
        """Initialise a new instance

        :arg Lx: horizontal extent of mesh
        :arg Ly: vertical extent of mesh
        """
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.vertices = np.asarray(
            [[0, 0], [self.Lx, 0], [0, self.Ly], [self.Lx, self.Ly]], dtype=float
        )
        self.cell2facet = [[0, 1, 2], [0, 3, 4]]
        self.facet2vertex = [[1, 2], [0, 2], [0, 1], [1, 3], [2, 3]]
        self.refine(nref)


class TriangleMesh(Mesh):
    """Trangular mesh ontained by refinement of a single triangle"""

    def __init__(self, corners=None, nref=0):
        """Initialise new instance

        :arg vertices: Coordinates of the three corners of the unrefined mesh
        :arg nref: number of refinements
        """
        super().__init__()
        corners = [[0, 0], [1, 0], [0, 1]] if corners is None else corners
        self.vertices = np.asarray(corners, dtype=float)
        self.cell2facet = [[0, 1, 2]]
        self.facet2vertex = [[1, 2], [0, 2], [0, 1]]
        self.refine(nref)
