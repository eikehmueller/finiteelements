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
    element = fs.finiteelement
    element_coord = fs_coord.finiteelement
    error_nrm_2 = 0
    for alpha in range(mesh.ncells):
        # local dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        # global dof-indices of coordinate field
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # local dof-indices of numerical solution
        ell = range(element.ndof)
        # global dof-indices of numerical solution
        ell_g = fs.local2global(alpha, ell)
        u_numerical_K = u_numerical.data[ell_g]
        zeta = np.asarray(quad.nodes)
        w_q = quad.weights
        T_coord = element_coord.tabulate(zeta)
        T_coord_partial = element_coord.tabulate_gradient(zeta)
        x_global = np.einsum("qla,l->aq", T_coord, x_dof_vector)
        u_exact_K = u_exact(x_global)
        T = element.tabulate(zeta)
        error_K = u_exact_K - T @ u_numerical_K
        jac = np.einsum("l,qlab->qab", x_dof_vector, T_coord_partial)
        error_nrm_2 += np.sum(w_q * error_K**2 * np.abs(np.linalg.det(jac)))
    return np.sqrt(error_nrm_2)
