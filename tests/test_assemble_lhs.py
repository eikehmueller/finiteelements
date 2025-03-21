import numpy as np
import pytest

from fem.polynomialelement import PolynomialElement
from fem.utilitymeshes import RectangleMesh
from fem.functionspace import FunctionSpace
from fem.quadrature import GaussLegendreQuadrature
from fem.algorithms import assemble_lhs


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_sparse_assembly(degree):
    """Check that assembled matrix is the same in dense and PETSc format"""
    element = PolynomialElement(degree)
    nref = 2
    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadrature(2 * degree)
    sparse_stiffness_matrix = (
        assemble_lhs(fs, quad, sparse=True).convert("dense").getDenseArray()
    )
    dense_stiffness_matrix = assemble_lhs(fs, quad, sparse=False)
    assert np.allclose(sparse_stiffness_matrix, dense_stiffness_matrix)
