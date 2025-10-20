"""Finite element assembly routines"""

import numpy as np
from petsc4py import PETSc

__all__ = [
    "assemble_rhs",
    "sparsity_lhs",
    "assemble_lhs",
]


def assemble_rhs(f, r, quad):
    """Assemble function into RHS

    :arg f: function, needs to be callable with f(x) where x is a 2-vector
    :arg r: dual function, must be an instance of DualFunction
    :arg quad: quadrature rule
    """
    fs = r.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    element = fs.finiteelement
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # local dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        # global dof-indices of coordinate field
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # local dof-indices of RHS vector
        ell = range(element.ndof)
        # global dof-indices of RHS vector
        ell_g = fs.local2global(alpha, ell)
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        zeta = np.asarray(quad.nodes)
        w_q = quad.weights
        f_X = f(np.dot(x_dof_vector, element_coord.tabulate(zeta)).T)
        T = element.tabulate(zeta)
        T_coord_partial = element_coord.tabulate_gradient(zeta)
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_partial)
        r.data[ell_g] += np.einsum("q,qi,q,q->i", w_q, T, f_X, np.abs(np.linalg.det(J)))


def sparsity_lhs(fs):
    """Sparsity structure for stiffness matrix

    :arg fs: function space
    """
    mesh = fs.mesh
    element = fs.finiteelement
    indices = [set() for _ in range(fs.ndof)]
    for cell in range(mesh.ncells):
        # global indices of function space
        J_global = fs.local2global(cell, range(element.ndof))
        for ell_g in J_global:
            indices[ell_g].update(set(J_global))
    row_start = [0]
    col_indices = []
    for row in indices:
        col_indices += sorted(row)
        row_start.append(len(col_indices))
    return row_start, col_indices


def assemble_lhs(fs, quad, kappa, omega):
    """Assemble LHS bilinear form into (dense) numpy matrix

    :arg fs: function space
    :arg quad: quadrature rule
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero order term
    """
    mesh = fs.mesh
    element = fs.finiteelement
    stiffness_matrix = np.zeros((fs.ndof, fs.ndof))
    fs_coord = mesh.coordinates.functionspace
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # global indices of function space
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        zeta = np.asarray(quad.nodes)
        w_q = quad.weights
        T_grad = element.tabulate_gradient(zeta)
        T = element.tabulate(zeta)
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        T_coord_partial = element_coord.tabulate_gradient(zeta)
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_partial)
        JT_J_inv = np.linalg.inv(np.einsum("qji,qjk->qik", J, J))
        local_matrix = kappa * np.einsum(
            "q,qjl,qlm,qkm,q->jk",
            w_q,
            T_grad,
            JT_J_inv,
            T_grad,
            np.abs(np.linalg.det(J)),
        ) + omega * np.einsum(
            "q,qj,qk,q->jk",
            w_q,
            T,
            T,
            np.abs(np.linalg.det(J)),
        )
        stiffness_matrix[np.ix_(ell_g, ell_g)] += local_matrix
    return stiffness_matrix


def assemble_lhs_sparse(fs, quad, kappa, omega):
    """Assemble LHS bilinear form into (sparse) PETSc matrix

    :arg fs: function space
    :arg quad: quadrature rule
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero order term
    """
    mesh = fs.mesh
    element = fs.finiteelement
    row_start, col_indices = sparsity_lhs(fs)
    stiffness_matrix = PETSc.Mat()
    stiffness_matrix.createAIJ((fs.ndof, fs.ndof), csr=(row_start, col_indices))
    fs_coord = mesh.coordinates.functionspace
    element_coord = fs_coord.finiteelement
    for alpha in range(mesh.ncells):
        # local dof-indices of function space
        ell = range(element.ndof)
        # global dof-indices of function space
        ell_g = fs.local2global(alpha, ell)
        ell_coord = range(element_coord.ndof)
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        zeta = np.asarray(quad.nodes)
        w_q = quad.weights
        T_grad = element.tabulate_gradient(zeta)
        T = element.tabulate(zeta)
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        T_coord_partial = element_coord.tabulate_gradient(zeta)
        J = np.einsum("l,qlab->qab", x_dof_vector, T_coord_partial)
        JT_J_inv = np.linalg.inv(np.einsum("qji,qjk->qik", J, J))
        local_matrix = kappa * np.einsum(
            "q,qjl,qlm,qkm,q->jk",
            w_q,
            T_grad,
            JT_J_inv,
            T_grad,
            np.abs(np.linalg.det(J)),
        ) + omega * np.einsum(
            "q,qj,qk,q->jk",
            w_q,
            T,
            T,
            np.abs(np.linalg.det(J)),
        )
        stiffness_matrix.setValues(ell_g, ell_g, local_matrix, addv=True)
    stiffness_matrix.assemble()
    return stiffness_matrix
