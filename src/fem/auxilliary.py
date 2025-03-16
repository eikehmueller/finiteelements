"""Some auxilliary methods"""

import numpy as np

__all__ = ["jacobian"]


def jacobian(self, mesh, cell, xi):
    """Calculate Jacobian in a given cell for quadrature points in reference triangle

    :arg mesh: underlying mesh
    :arg cell: index of mesh cell
    :arg xi: point in reference cell at which the Jacobian is to be evaluated
    """
    fs = mesh.coordinates.functionspace
    element = fs.finiteelement
    gradient = element.tabulate_gradient(xi)
    j_g = fs.local2global(cell, range(element.ndof))
    return np.dot(self.coordinates.data[j_g], gradient)
