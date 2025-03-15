import pytest
from fem.utilitymeshes import RectangleMesh, TriangleMesh
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_local2global_triangle(degree):
    """Check that local to global indexing is correct on unrefined triangle mesh"""
    mesh = TriangleMesh(nref=0)
    element = PolynomialElement(degree)
    functionspace = FunctionSpace(mesh, element)
    idx = range(element.ndof)
    assert functionspace.local2global(0, idx) == list(idx)


def test_local2global_rectangle():
    """Check that local to global indexing is correct on unrefined rectangle mesh"""
    mesh = RectangleMesh(nref=0)
    element = PolynomialElement(3)
    functionspace = FunctionSpace(mesh, element)
    idx = range(10)
    assert (functionspace.local2global(0, idx) == [0, 1, 2, 4, 5, 6, 7, 8, 9, 14]) and (
        functionspace.local2global(1, idx) == [0, 3, 2, 10, 11, 6, 7, 12, 13, 15]
    )
