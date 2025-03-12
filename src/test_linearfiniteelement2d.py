import numpy as np
from finiteelement2d import LinearFiniteElement2d


def test_linear_finiteelement2d_nodal_evaluations():
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    basis = LinearFiniteElement2d()
    xi = np.asarray([[0, 0], [1, 0], [0, 1]])
    evaluations = np.empty((3, 3))
    for j in range(3):
        evaluations[j, :] = basis.evaluate(xi[j, :])
    assert np.allclose(
        evaluations,
        np.eye(3),
    )


def test_linear_finiteelement2d_cell2dof_map():
    """Check that the cell2dof map is correct for linear basis functions"""
    basis = LinearFiniteElement2d()
    assert basis.cell2dof == []


def test_linear_finiteelement2d_facet2dof_map():
    """Check that the facet2dof map is correct for linear basis functions"""
    basis = LinearFiniteElement2d()
    assert basis.facet2dof == []


def test_linear_finiteelement2d_vertex2dof_map():
    """Check that the vertex2dof map is correct for linear basis functions"""
    basis = LinearFiniteElement2d()
    assert basis.vertex2dof == [[0], [1], [2]]
