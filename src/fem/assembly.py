"""Finite element assembly routines for reaction- diffusion equation"""

import numpy as np
from petsc4py import PETSc

__all__ = [
    "assemble_rhs",
    "sparsity_lhs",
    "assemble_lhs",
    "assemble_lhs_sparse",
]


def assemble_rhs(f, r, quad):
    """Assemble right hand as a CoFunction

    The linear form int_Omega (f,.) dx is assembled into the dual function r
    for a given function f.

    :arg f: function, needs to be callable with f(x) where x is a 2-vector
            or a rank 2 tensor of shape (2,n)
    :arg r: dual function which will hold the result,
            must be an instance of DualFunction
    :arg quad: quadrature rule
    """
    fs = r.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    # finite element of function space
    element = fs.finiteelement
    # finite element of coordinate function space
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # local and global dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # local and global dof-indices of function space
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        # local dof-vector of coordinates
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # quadrature points and weights
        zeta_q = np.asarray(quad.nodes)
        w_q = quad.weights
        # vector with evaluation of function f at the quadrature points
        f_q = f(np.dot(x_dof_vector, element_coord.tabulate(zeta_q)).T)
        # tabulation of basis functions at quadrature points
        T = element.tabulate(zeta_q)
        # tabulation of gradient of coordinate basis functions at quadrature points
        T_coord_grad = element_coord.tabulate_gradient(zeta_q)
        # Jacobian and its determinant
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_grad)
        det_J = np.abs(np.linalg.det(J))
        r.data[ell_g] += np.einsum("q,ql,q,q->l", w_q, T, f_q, det_J)


def sparsity_lhs(fs):
    """Sparsity structure for finite element stiffness matrix

    Returns the row-start and column-indices arrays that can be used to
    construct the sparse matrix is CSR format. This assumes that the unknowns in
    each cell couple to unknowns in the same cell and its direct neighbours.

    :arg fs: function space
    """
    mesh = fs.mesh
    # finite element of function space
    element = fs.finiteelement
    # list of sets, which will hold the global indices for each non-zero
    # entry in a particular row of the matrix
    indices = [set() for _ in range(fs.ndof)]
    for alpha in range(mesh.ncells):
        # global indices of function space
        J_global = fs.local2global(alpha, range(element.ndof))
        for ell_g in J_global:
            indices[ell_g].update(set(J_global))
    # construct row-pointer and column-index arrays
    row_start = [0]
    col_indices = []
    for row in indices:
        col_indices += sorted(row)
        row_start.append(len(col_indices))
    return row_start, col_indices


def assemble_lhs(fs, quad, kappa, omega):
    """Assemble left hand side bilinear form into (dense) numpy matrix

    The problem considered here is the weak formulation of the
    reaction-diffusion equation

        - div (kappa grad(u)) + omega u = f

    :arg fs: function space
    :arg quad: quadrature rule
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero order term
    """
    mesh = fs.mesh
    stiffness_matrix = np.zeros((fs.ndof, fs.ndof))
    fs_coord = mesh.coordinates.functionspace
    # finite element of function space
    element = fs.finiteelement
    # finite element of coordinate function space
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # local and global indices of function space
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        # local and global indices of coordinate field
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # quadrature points and weights
        zeta_q = np.asarray(quad.nodes)
        w_q = quad.weights
        # tabulation of basis functions and their gradients at quadrature points
        T = element.tabulate(zeta_q)
        T_grad = element.tabulate_gradient(zeta_q)
        # tabulation of gradient of coordinate basis functions at quadrature points
        T_coord_grad = element_coord.tabulate_gradient(zeta_q)
        # local coordinate vector
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # Jacobian J and matrix (J^T J)^{-1}
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_grad)
        JT_J_inv = np.linalg.inv(np.einsum("qca,qcb->qab", J, J))
        # local stiffness matrix
        local_matrix = kappa * np.einsum(
            "q,qka,qab,qlb,q->lk",
            w_q,
            T_grad,
            JT_J_inv,
            T_grad,
            np.abs(np.linalg.det(J)),
        ) + omega * np.einsum(
            "q,qk,ql,q->lk",
            w_q,
            T,
            T,
            np.abs(np.linalg.det(J)),
        )
        # Insert into global stiffness matrix
        stiffness_matrix[np.ix_(ell_g, ell_g)] += local_matrix
    return stiffness_matrix


def assemble_lhs_sparse(fs, quad, kappa, omega):
    """Assemble left hand side bilinear form into (sparse) PETSc matrix

    The problem considered here is the weak formulation of the
    reaction-diffusion equation

        - div (kappa grad(u)) + omega u = f

    :arg fs: function space
    :arg quad: quadrature rule
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero order term
    """
    mesh = fs.mesh
    # Construct PETSc matrix in CSR format
    row_start, col_indices = sparsity_lhs(fs)
    stiffness_matrix = PETSc.Mat()
    stiffness_matrix.createAIJ((fs.ndof, fs.ndof), csr=(row_start, col_indices))
    fs_coord = mesh.coordinates.functionspace
    # finite element of function space
    element = fs.finiteelement
    # finite element of coordinate function space
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # local and global dof-indices of function space
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        # local and global dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # quadrature points and weights
        zeta_q = np.asarray(quad.nodes)
        w_q = quad.weights
        # Tabulation of finite element basis functions and their gradients at quadrature points
        T = element.tabulate(zeta_q)
        T_grad = element.tabulate_gradient(zeta_q)
        # Tabulation of gradient of coordinate basis functions
        T_coord_partial = element_coord.tabulate_gradient(zeta_q)
        # Local dof-vector of coordinate field
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        # Jacobian J and matrix (J^T J)^{-1}
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_partial)
        JT_J_inv = np.linalg.inv(np.einsum("qji,qjk->qik", J, J))
        # local stiffness matrix
        local_matrix = kappa * np.einsum(
            "q,qka,qab,qlb,q->lk",
            w_q,
            T_grad,
            JT_J_inv,
            T_grad,
            np.abs(np.linalg.det(J)),
        ) + omega * np.einsum(
            "q,qk,ql,q->lk",
            w_q,
            T,
            T,
            np.abs(np.linalg.det(J)),
        )
        # insert values in global PETSc matrix
        stiffness_matrix.setValues(ell_g, ell_g, local_matrix, addv=True)
    # Don't forget to call assemble() at the very end
    stiffness_matrix.assemble()
    return stiffness_matrix
