"""Test suite for piecewise polynomial finite element"""

import numpy as np
import pytest
from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement


@pytest.fixture
def rng():
    """Random number generator"""
    return np.random.default_rng(seed=2895167)


def nodal_points(degree):
    """Return locations of nodal points on reference triaggle"""
    xi = []
    h = 1 / degree
    for b in range(degree + 1):
        for a in range(degree + 1 - b):
            xi.append([a * h, b * h])
    xi = np.asarray(xi)
    if degree == 1:
        perm = [0, 1, 2]
    elif degree == 2:
        perm = [0, 5, 1, 4, 3, 2]
    elif degree == 3:
        perm = [0, 7, 8, 1, 6, 9, 3, 5, 4, 2]
    elif degree == 4:
        perm = [0, 9, 10, 11, 1, 8, 12, 13, 3, 7, 14, 4, 6, 5, 2]
    else:
        raise RuntimeError(
            "Nodal points only available for polynomial degree between 1 and 4"
        )
    return xi[np.argsort(perm)]


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomialelement_nodal_tabulation(degree):
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    element = PolynomialElement(degree)
    xi = nodal_points(degree)
    evaluations = np.empty((element.ndof, element.ndof))
    for j in range(element.ndof):
        evaluations[j, :] = element.tabulate(xi[j, :])
    assert np.allclose(evaluations, np.eye(element.ndof))


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomialelement_dof_tabulation(degree):
    """Check that dof-evaluation works as expected"""
    element = PolynomialElement(degree)
    # function to test
    fhat = lambda x: np.exp(0.5 + x[0] + 2 * x[1])
    xi = nodal_points(degree)
    assert np.allclose(fhat(xi.T), element.tabulate_dofs(fhat))


def test_polynomialelement_agrees_with_linear(rng):
    """Check that tabulation of polynomial basis functions of degree 1 give same
    result as for linear basis functions"""
    element_lin = LinearElement()
    element_poly = PolynomialElement(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(
            element_lin.tabulate(xi[j, :]), element_poly.tabulate(xi[j, :])
        )


def test_polynomialelement_agrees_with_linear_gradient(rng):
    """Check that gradient tabulation of polynomial basis functions of degree 1 give same
    result as for linear basis functions"""
    element_lin = LinearElement()
    element_poly = PolynomialElement(1)
    nsamples = 32
    xi = rng.uniform(size=(nsamples, 2))
    for j in range(nsamples):
        assert np.allclose(
            element_lin.tabulate_gradient(xi[j, :]),
            element_poly.tabulate_gradient(xi[j, :]),
        )
