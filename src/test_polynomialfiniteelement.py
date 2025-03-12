import numpy as np
import pytest
from finiteelement import LinearFiniteElement2d, PolynomialFiniteElement2d


@pytest.fixture
def rng():
    """Random number generator"""
    return np.random.default_rng(seed=2895167)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomial_finiteelement2d_nodal_evaluations(degree):
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    element = PolynomialFiniteElement2d(degree)
    xi = []
    h = 1 / degree
    for b in range(degree + 1):
        for a in range(degree + 1 - b):
            xi.append([a * h, b * h])
    xi = np.asarray(xi)
    evaluations = np.empty((element.ndof, element.ndof))
    for j in range(element.ndof):
        evaluations[j, :] = element.evaluate(xi[j, :])
    expected = np.eye(element.ndof)
    if degree == 2:
        perm = [0, 5, 1, 4, 3, 2]
    elif degree == 3:
        perm = [0, 7, 8, 1, 6, 9, 3, 5, 4, 2]
    elif degree == 4:
        perm = [0, 9, 10, 11, 1, 8, 12, 13, 3, 7, 14, 4, 6, 5, 2]
    else:
        perm = range(element.ndof)
    expected = expected[perm, :]
    assert np.allclose(evaluations, expected)


def test_polynomial_finiteelement2d_linear(rng):
    """Check that evaluation of polynomial basis functions of degree 1 give same
    result as linear basis functions"""

    element_lin = LinearFiniteElement2d()
    element_poly = PolynomialFiniteElement2d(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(
            element_lin.evaluate(xi[j, :]), element_poly.evaluate(xi[j, :])
        )


def test_polynomial_finiteelement2d_linear_gradient(rng):
    """Check that gradient evaluation of polynomial basis functions of degree 1 give same
    result as linear basis functions"""
    element_lin = LinearFiniteElement2d()
    element_poly = PolynomialFiniteElement2d(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(
            element_lin.evaluate_gradient(xi[j, :]),
            element_poly.evaluate_gradient(xi[j, :]),
        )
