import numpy as np
import pytest
from finiteelement import LinearFiniteElement2d, PolynomialFiniteElement2d


@pytest.fixture
def rng():
    return np.random.default_rng(seed=2895167)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomial_finiteelement2d_nodal_evaluations(degree):
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    basis = PolynomialFiniteElement2d(degree)
    xi = []
    h = 1 / degree
    for b in range(degree + 1):
        for a in range(degree + 1 - b):
            xi.append([a * h, b * h])
    xi = np.asarray(xi)
    evaluations = np.empty((basis.ndof, basis.ndof))
    for j in range(basis.ndof):
        evaluations[j, :] = basis.evaluate(xi[j, :])

    assert np.allclose(
        evaluations,
        np.eye(basis.ndof),
    )


def test_polynomial_finiteelement2d_linear(rng):
    """Check that evaluation of polynomial basis functions of degree 1 give same
    result as linear basis functions"""

    basis_lin = LinearFiniteElement2d()
    basis_poly = PolynomialFiniteElement2d(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(basis_lin.evaluate(xi[j, :]), basis_poly.evaluate(xi[j, :]))


def test_polynomial_finiteelement2d_linear_gradient(rng):
    """Check that gradient evaluation of polynomial basis functions of degree 1 give same
    result as linear basis functions"""
    basis_lin = LinearFiniteElement2d()
    basis_poly = PolynomialFiniteElement2d(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(
            basis_lin.evaluate_gradient(xi[j, :]),
            basis_poly.evaluate_gradient(xi[j, :]),
        )


def test_polynomial_finiteelement2d_cell2dof_map():
    """Check that the cell2dof map is correct for p=4 basis functions"""
    basis = PolynomialFiniteElement2d(4)
    assert basis.cell2dof == [6, 7, 10]


def test_polynomial_finiteelement2d_facet2dof_map():
    """Check that the facet2dof map is correct for p=4 basis functions"""
    basis = PolynomialFiniteElement2d(4)
    assert basis.facet2dof == [[8, 11, 13], [12, 9, 5], [1, 2, 3]]


def test_polynomial_finiteelement2d_vertex2dof_map():
    """Check that the vertex2dof map is correct for p=4 basis functions"""
    basis = PolynomialFiniteElement2d(4)
    assert basis.vertex2dof == [[0], [4], [14]]
