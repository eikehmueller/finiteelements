"""Test suite for piecewise linear finite elements"""

import numpy as np
import pytest
from fem.linearelement import LinearElement


@pytest.fixture
def element():
    return LinearElement()


def test_ndof_per_vertex(element):
    """Check that the number of unknowns per vertex is correct"""
    assert element.ndof_per_vertex == 1


def test_ndof_per_facet(element):
    """Check that the number of unknowns per facet is correct"""
    element = LinearElement()
    assert element.ndof_per_facet == 0


def test_ndof_per_interior(element):
    """Check that the number of unknowns per interior is correct"""
    assert element.ndof_per_interior == 0


def test_nodal_tabulation(element):
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    xi = np.asarray([[0, 0], [1, 0], [0, 1]])
    evaluations = np.empty((3, 3))
    for j in range(3):
        evaluations[j, :] = element.tabulate(xi[j, :])
    assert np.allclose(
        evaluations,
        np.eye(3),
    )


def test_nodal_tabulation_vectorised(element):
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    xi = np.asarray([[0, 0], [1, 0], [0, 1]])
    assert np.allclose(
        element.tabulate(xi),
        np.eye(3),
    )


def test_gradient_tabulation(element):
    """Check that the gradient is correctly tabulated"""
    xi = np.asarray([0.2, 0.3])
    assert np.allclose(element.tabulate_gradient(xi), [[-1, -1], [1, 0], [0, 1]])


def test_gradient_tabulation_vectorised(element):
    """Check that the gradient is correctly tabulated"""
    xi = np.asarray([[0.2, 0.3], [0.1, 0.6]])
    assert np.allclose(
        element.tabulate_gradient(xi),
        [[[-1, -1], [1, 0], [0, 1]], [[-1, -1], [1, 0], [0, 1]]],
    )


def test_linearelement_dof_tabulation(element):
    """Check that dof-evaluation works as expected"""
    # function to test
    fhat = lambda x: np.exp(0.5 + x[..., 0] + 2 * x[..., 1])
    nodal_points = np.asarray([[0, 0], [1, 0], [0, 1]])
    assert np.allclose(fhat(nodal_points), element.tabulate_dofs(fhat))


def test_dofmap(element):
    """Check that dof-map works as expected"""
    indices = [
        element.dofmap(*element.inverse_dofmap(ell)) for ell in range(element.ndof)
    ]
    assert np.allclose(indices, list(range(element.ndof)))
