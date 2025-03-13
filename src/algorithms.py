import numpy as np
from function import Function


def interpolate(f, u):
    """Interpolate a function to a finite element function

    :arg f: function, needs to be callable with f(x) where x is a 2-vector
    :arg u: finite element function
    """
    fs = u.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates_x.functionspace
    element = fs.finiteelement
    element_coord = fs_coord.finiteelement
    for cell in range(mesh.ncells):
        X = np.empty((3, 2))
        for j in range(3):
            j_g = fs_coord.local2global(cell, j)
            X[j, 0] = mesh.coordinates_x.data[j_g]
            X[j, 1] = mesh.coordinates_y.data[j_g]
        f_hat = lambda xhat: f(np.dot(element_coord.evaluate(xhat), X))
        dof_vector = element.dofs(f_hat)
        for j in range(element.ndof):
            j_g = fs.local2global(cell, j)
            u.data[j_g] = dof_vector[j]
