"""Computation of finite element L2 error norm"""

import numpy as np

__all__ = ["error_norm"]


def error_norm(u_numerical, u_exact, quad):
    """Compute L2 error norm ||u-u_{exact}||_{L_2(Omega)}

    :arg u_numerical: numerical solution, must be an instance of Function
    :arg u_exact: function, must be callable everywhere in the domain
    :arg quad: quadrature rule
    """
    fs = u_numerical.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    # finite element of function space
    element = fs.finiteelement
    # finite element of coordinate field
    element_coord = fs_coord.finiteelement
    error_nrm_2 = 0
    for alpha in range(mesh.ncells):
        # local and global dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # local and global dof-indices of numerical solution
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        # local dof-vector of coordinate field
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # local dof-vector of numerical solution
        u_numerical_dof_vector = u_numerical.data[ell_g]
        # quadrature points and weights
        zeta_q = np.asarray(quad.nodes)
        w_q = quad.weights
        # tabulation of finite element basis functions at quadrature points
        T = element.tabulate(zeta_q)
        # tabulation of coordinate basis functions and their gradients at quadrature points
        T_coord = element_coord.tabulate(zeta_q)
        T_coord_grad = element_coord.tabulate_gradient(zeta_q)
        # physical coordinates at quadrature point
        x_q = np.einsum("qla,l->qa", T_coord, x_dof_vector)
        # evaluation of exact solution at quadrature points
        u_exact_q = u_exact(x_q.T)
        # error at quadrature points
        error_q = u_exact_q - T @ u_numerical_dof_vector
        # Jacobian and its determinant
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_grad)
        det_J = np.abs(np.linalg.det(J))
        # Increment square of error
        error_nrm_2 += np.sum(w_q * error_q**2 * det_J)
    return np.sqrt(error_nrm_2)
