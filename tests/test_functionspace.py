import pytest
from fem.utilitymeshes import RectangleMesh, TriangleMesh
from fem.functionspace import FunctionSpace
from fixtures import polynomial_element, element


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_local2global_triangle(degree, element):
    """Check that local to global indexing is correct on unrefined triangle mesh"""
    mesh = TriangleMesh(nref=0)
    functionspace = FunctionSpace(mesh, element)
    idx = list(range(element.ndof))
    idx[3 + (degree - 1) : 3 + 2 * (degree - 1)] = idx[
        2 + 2 * (degree - 1) : 2 + (degree - 1) : -1
    ]
    assert functionspace.local2global(0, range(element.ndof)) == list(idx)


def test_local2global_rectangle():
    """Check that local to global indexing is correct on unrefined rectangle mesh"""
    mesh = RectangleMesh(nref=0)
    element = polynomial_element(3)
    functionspace = FunctionSpace(mesh, element)
    idx = range(10)
    print(functionspace.local2global(0, idx))
    print(functionspace.local2global(1, idx))
    assert (functionspace.local2global(0, idx) == [0, 1, 2, 4, 5, 7, 6, 8, 9, 14]) and (
        functionspace.local2global(1, idx) == [3, 2, 1, 5, 4, 10, 11, 13, 12, 15]
    )
