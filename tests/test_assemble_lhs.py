import numpy as np
import pytest

import petsc4py

petsc4py.init()

from fem.polynomialelement import PolynomialElement
from fem.utilitymeshes import RectangleMesh, TriangleMesh
from fem.functionspace import FunctionSpace
from fem.quadrature import GaussLegendreQuadrature
from fem.algorithms import assemble_lhs
from fem.auxilliary import csr_as_dense


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_sparse_assembly(degree):
    """Check that assembled matrix is the same in dense and PETSc format"""
    element = PolynomialElement(degree)
    nref = 0
    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    mesh = TriangleMesh(nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadrature(2 * degree)

    sparse_stiffness_matrix = assemble_lhs(fs, quad, sparse=True)
    dense_stiffness_matrix = assemble_lhs(fs, quad, sparse=False)
    assert np.allclose(csr_as_dense(sparse_stiffness_matrix), dense_stiffness_matrix)
