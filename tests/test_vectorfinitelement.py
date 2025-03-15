import pytest
import numpy as np
from finiteelement import PolynomialFiniteElement2d, VectorFiniteElement2d


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
def test_vector_finiteelement2d_tabulation(degree):
    finiteelement = PolynomialFiniteElement2d(degree)
    vectorelement = VectorFiniteElement2d(finiteelement)
    xi = nodal_points(degree)
    tabulation = np.empty((finiteelement.ndof, vectorelement.ndof, 2))
    expected_tabulation = np.zeros((finiteelement.ndof, vectorelement.ndof, 2))
    for j in range(finiteelement.ndof):
        tabulation[j, :, :] = vectorelement.tabulate(xi[j])
        expected_tabulation[j, 2 * j : 2 * j + 2, :] = np.eye(2)
    assert np.allclose(tabulation, expected_tabulation)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_vector_finiteelement2d_dof_tabulation(degree):
    finiteelement = PolynomialFiniteElement2d(degree)
    vectorelement = VectorFiniteElement2d(finiteelement)
    xi = nodal_points(degree)
    fhat = lambda x: np.asarray([np.exp(x[0]) * (1 + x[1]), x[0] + np.exp(2 * x[1])])
    assert np.allclose(vectorelement.tabulate_dofs(fhat), fhat(xi.T).T.flatten())
