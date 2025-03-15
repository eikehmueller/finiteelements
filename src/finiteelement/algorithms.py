import numpy as np


def interpolate(f, u):
    """Interpolate a function to a finite element function

    :arg f: function, needs to be callable with f(x) where x is a 2-vector
    :arg u: finite element function
    """
    fs = u.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    element = fs.finiteelement
    element_coord = fs_coord.finiteelement
    for cell in range(mesh.ncells):
        j_g_coord = fs_coord.local2global(cell, range(element_coord.ndof))
        x_dof_vector = mesh.coordinates.data[j_g_coord]
        f_hat = lambda xhat: f(np.dot(x_dof_vector, element_coord.tabulate(xhat)))
        u_dof_vector = element.tabulate_dofs(f_hat)
        j_g_u = fs.local2global(cell, range(element.ndof))
        u.data[j_g_u] = u_dof_vector[:]
