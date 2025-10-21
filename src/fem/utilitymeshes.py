"""Factory methods for commonly used meshes"""

import numpy as np
from fem.mesh import Mesh

__all__ = ["rectangle_mesh", "triangle_mesh"]


def rectangle_mesh(Lx=1.0, Ly=1.0, nref=0):
    """Factory method for triangular mesh obtained by refinement of single rectangle

    :arg Lx: horizontal extent of mesh
    :arg Ly: vertical extent of mesh
    """
    vertices = np.asarray([[0, 0], [Lx, 0], [0, Ly], [Lx, Ly]], dtype=float)
    cell2facet = [[0, 1, 2], [0, 3, 4]]
    facet2vertex = [[1, 2], [0, 2], [0, 1], [1, 3], [2, 3]]
    mesh = Mesh(vertices, cell2facet, facet2vertex)
    mesh.refine(nref)
    return mesh


def triangle_mesh(corners=None, nref=0):
    """Factory methdo for triangular mesh ontained by refinement of a single triangle

    :arg vertices: coordinates of the three corners of the unrefined mesh
    :arg nref: number of refinements
    """

    vertices = np.asarray(
        [[0, 0], [1, 0], [0, 1]] if corners is None else np.asarray(corners),
        dtype=float,
    )
    cell2facet = [[0, 1, 2]]
    facet2vertex = [[1, 2], [0, 2], [0, 1]]
    mesh = Mesh(vertices, cell2facet, facet2vertex)
    mesh.refine(nref)
    return mesh
