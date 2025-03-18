import numpy as np

from fem.auxilliary import jacobian


def interpolate(f, u):
    """Interpolate a function to a finite element function

    :arg f: function, needs to be callable with f(x) where x is a 2-vector
    :arg u: finite element function, instance of Function
    """
    fs = u.functionspace
    mesh = fs.mesh
    fs_coord = mesh.coordinates.functionspace
    element = fs.finiteelement
    element_coord = fs_coord.finiteelement
    for cell in range(mesh.ncells):
        j_g_coord = fs_coord.local2global(cell, range(element_coord.ndof))
        x_dof_vector = mesh.coordinates.data[j_g_coord]

        # local function
        f_hat = lambda x_hat, x_dof_vector=x_dof_vector: f(
            np.einsum("j,ijk->ik", x_dof_vector, element_coord.tabulate(x_hat.T)).T
        )

        u_dof_vector = element.tabulate_dofs(f_hat)
        j_g_u = fs.local2global(cell, range(element.ndof))
        u.data[j_g_u] = u_dof_vector[:]


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
    for cell in range(mesh.ncells):
        # global indices of coordinate field
        j_g_coord = fs_coord.local2global(cell, range(element_coord.ndof))
        # global indices of RHS
        j_g_r = fs.local2global(cell, range(element.ndof))
        x_dof_vector = mesh.coordinates.data[j_g_coord]
        x_q_hat = np.asarray(quad.nodes)
        w_q = quad.weights
        f_X = f(np.dot(x_dof_vector, element_coord.tabulate(x_q_hat)).T)
        phi = element.tabulate(x_q_hat)
        J = jacobian(mesh, cell, x_q_hat)
        r.data[j_g_r] += np.einsum(
            "q,qi,q,q->i", w_q, phi, f_X, np.abs(np.linalg.det(J))
        )


def assemble_lhs(fs, quad):
    """Assemble LHS bilinear form into matrix

    :arg fs: function space
    :arg quad: quadrature rule
    """
    mesh = fs.mesh
    element = fs.finiteelement
    stiffness_matrix = np.zeros((fs.ndof, fs.ndof))
    for cell in range(mesh.ncells):
        # global indices of function space
        j_g = fs.local2global(cell, range(element.ndof))
        x_q_hat = np.asarray(quad.nodes)
        w_q = quad.weights
        grad_phi = element.tabulate_gradient(x_q_hat)
        phi = element.tabulate(x_q_hat)
        J = jacobian(mesh, cell, x_q_hat)
        JT_J_inv = np.linalg.inv(np.einsum("qji,qjk->qik", J, J))
        local_matrix = np.einsum(
            "q,qjl,qlm,qkm,q->jk",
            w_q,
            grad_phi,
            JT_J_inv,
            grad_phi,
            np.abs(np.linalg.det(J)),
        ) + np.einsum(
            "q,qj,qk,q->jk",
            w_q,
            phi,
            phi,
            np.abs(np.linalg.det(J)),
        )
        stiffness_matrix[np.ix_(j_g, j_g)] += local_matrix
    return stiffness_matrix
