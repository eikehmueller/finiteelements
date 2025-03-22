"""Some auxilliary methods"""

import numpy as np

__all__ = ["jacobian", "csr_as_dense"]


def jacobian(mesh, cell, xi):
    """Calculate Jacobian in a given cell for quadrature points in reference triangle

    :arg mesh: underlying mesh
    :arg cell: index of mesh cell
    :arg xi: point in reference cell at which the Jacobian is to be evaluated
    """
    fs = mesh.coordinates.functionspace
    element = fs.finiteelement
    gradient = element.tabulate_gradient(xi)
    j_g = fs.local2global(cell, range(element.ndof))
    return np.einsum("j,...jkl->...kl", mesh.coordinates.data[j_g], gradient)


def csr_as_dense(mat):
    """Convert PETSc matrix stored in CSR format to a dense matrix

    :arg mat:PETSc matrix to be converted
    """
    indexptr, indices, data = mat.getValuesCSR()
    mat_dense = np.zeros(mat.getSize())
    k = 0
    for i in range(len(indexptr) - 1):
        for j in range(indexptr[i], indexptr[i + 1]):
            mat_dense[i, indices[j]] = data[k]
            k += 1
    return mat_dense
