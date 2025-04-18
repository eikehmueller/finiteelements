import numpy as np
from petsc4py import PETSc
from fem.quadrature import (
    GaussLegendreQuadratureLineSegment,
    GaussLegendreQuadratureReferenceTriangle,
)

__all__ = [
    "assemble_lhs",
    "assemble_rhs",
    "two_norm",
]


def assemble_lhs(element, n_q, kappa, omega):
    """Assemble LHS bilinear form into matrix

    :arg element: finite element
    :arg n_q: number of quadrature points
    :arg kappa: coefficient kappa of diffusion term
    :arg omega: coefficient kappa of zero-order term
    """
    quad = GaussLegendreQuadratureReferenceTriangle(n_q)
    zeta_q = np.asarray(quad.nodes)
    w_q = quad.weights
    grad_phi = element.tabulate_gradient(zeta_q)
    phi = element.tabulate(zeta_q)
    stiffness_matrix = kappa * np.einsum(
        "q,qik,qjk->ij", w_q, grad_phi, grad_phi
    ) + omega * np.einsum(
        "q,qi,qj->ij",
        w_q,
        phi,
        phi,
    )
    return stiffness_matrix


def assemble_rhs(f, g, element, n_q):
    """Assemble functions into RHS on unit triangle

    :arg f: RHS function, needs to be callable with f(x) where x is a 2-vector
    :arg g: Neumann boundary function, needs to be callable with f(x) where x is a 2-vector
    :arg element: finite element
    :arg n_q: number of quadrature points
    """
    quad = GaussLegendreQuadratureReferenceTriangle(n_q)
    zeta_q = np.asarray(quad.nodes)
    w_q = quad.weights
    f_q = f(zeta_q)
    phi = element.tabulate(zeta_q)
    r = np.einsum("q,q,qi->i", w_q, f_q, phi)
    for v_a, v_b in [[(1, 0), (0, 1)], [(0, 1), (0, 0)], [(0, 0), (1, 0)]]:
        quad_facet = GaussLegendreQuadratureLineSegment(v_a, v_b, n_q)
        w_facet_q = quad_facet.weights
        zeta_facet_q = np.asarray(quad_facet.nodes)
        g_q = g(zeta_facet_q)
        phi_facet = element.tabulate(zeta_facet_q)
        r += np.einsum("q,q,qi->i", w_facet_q, g_q, phi_facet)
    return r


def two_norm(w, element, n_q):
    """Compute L2 norm of a function

    :arg w: dof-vector of function on reference triangle
    :arg n_q: number of quadrature points
    """
    quad = GaussLegendreQuadratureReferenceTriangle(n_q)
    x_q_hat = np.asarray(quad.nodes)
    w_q = quad.weights
    phi = element.tabulate(x_q_hat)

    nrm2 = np.einsum(
        "q,j,k,qj,qk->",
        w_q,
        w,
        w,
        phi,
        phi,
    )
    return np.sqrt(nrm2)
