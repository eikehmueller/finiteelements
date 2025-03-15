from fem.utilitymeshes import RectangleMesh
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace


def test_local2global():
    mesh = RectangleMesh(nref=0)
    element = PolynomialElement(3)
    functionspace = FunctionSpace(mesh, element)
    idx = range(9)
    print(functionspace.local2global(0, idx))
