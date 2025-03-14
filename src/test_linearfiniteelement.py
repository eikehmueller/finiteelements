import numpy as np
from finiteelement import LinearFiniteElement2d


def test_linear_finiteelement2d_nodal_evaluations():
    """Check that phi_k(xi_j) = delta_{j,k} for the linear basis functions"""
    element = LinearFiniteElement2d()
    xi = np.asarray([[0, 0], [1, 0], [0, 1]])
    evaluations = np.empty((3, 3))
    for j in range(3):
        evaluations[j, :] = element.tabulate(xi[j, :])
    assert np.allclose(
        evaluations,
        np.eye(3),
    )
