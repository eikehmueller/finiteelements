"""Some auxilliary methods"""

import numpy as np

__all__ = ["jacobian"]


def jacobian(mesh, alpha, zeta):
    """Calculate Jacobian in a given cell for quadrature points in reference triangle

    :arg mesh: underlying mesh
    :arg alpha: index of mesh cell
    :arg zeta: point in reference cell at which the Jacobian is to be evaluated
        can also be a list of n points, passed as an array of shape (n,2)
    """
    fs = mesh.coordinates.functionspace
    element = fs.finiteelement
    # tabulation of gradients of coordinate basis functions
    T_partial = element.tabulate_gradient(zeta)
    # local dof indices
    ell = range(element.ndof)
    # global dof-indices
    ell_global = fs.local2global(alpha, ell)
    # dofs of local coordinates
    X_local = mesh.coordinates.data[ell_global]
    return np.einsum("j,...jkl->...kl", X_local, T_partial)
