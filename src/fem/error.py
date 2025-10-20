"""Computation of finite element error norm"""

import numpy as np

__all__ = ["error_norm"]


def error_norm(u_numerical, u_exact, quad):
    """Compute L2 error norm

    :arg u_numerical: numerical solution, must be an instance of Function
    :arg u_exact: function, must be callable everywhere in the domain
    :arg quad: quadrature rule
    """
    fs = u_numerical.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    # Finite element of function space
    element = fs.finiteelement
    # Finite element of coordinate field
    element_coord = fs_coord.finiteelement
    error_nrm_2 = 0
    for alpha in range(mesh.ncells):
        # local and global dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # local and global dof-indices of numerical solution
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        # Local dof-vector of coordinate field
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # Local dof-vector of numerical solution
        u_numerical_dof_vector = u_numerical.data[ell_g]
        # Quadrature points and weights
        zeta_q = np.asarray(quad.nodes)
        w_q = quad.weights
        # Tabulation of finite element basis functions at quadrature points
        T = element.tabulate(zeta_q)
        # Tabulation of coordinate basis functions and their gradients at quadrature points
        T_coord = element_coord.tabulate(zeta_q)
        T_coord_grad = element_coord.tabulate_gradient(zeta_q)
        # Physical coordinates at quadrature point
        x_q = np.einsum("qla,l->aq", T_coord, x_dof_vector)
        # Evaluation of exact solution at quadrature points
        u_exact_q = u_exact(x_q)
        # Error at quadrature points
        error_q = u_exact_q - T @ u_numerical_dof_vector
        jac = np.einsum("l,qlab->qab", x_dof_vector, T_coord_grad)
        # Increment square of error
        error_nrm_2 += np.sum(w_q * error_q**2 * np.abs(np.linalg.det(jac)))
    return np.sqrt(error_nrm_2)
