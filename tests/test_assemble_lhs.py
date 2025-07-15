import numpy as np
import pytest


from fem.polynomialelement import PolynomialElement
from fem.utilitymeshes import RectangleMesh, TriangleMesh
from fem.functionspace import FunctionSpace
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle
from fem.algorithms import assemble_lhs, assemble_lhs_sparse
from fem.auxilliary import csr_as_dense


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_sparse_assembly(degree):
    """Check that assembled matrix is the same in dense and PETSc format"""
    kappa = 0.9
    omega = 0.4
    element = PolynomialElement(degree)
    nref = 0
    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    mesh = TriangleMesh(nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadratureReferenceTriangle(2 * degree)

    sparse_stiffness_matrix = assemble_lhs_sparse(fs, quad, kappa, omega)
    dense_stiffness_matrix = assemble_lhs(fs, quad, kappa, omega)
    assert np.allclose(csr_as_dense(sparse_stiffness_matrix), dense_stiffness_matrix)
