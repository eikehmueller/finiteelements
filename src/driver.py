"""Main program"""

import numpy as np

from mesh import RectangleMesh
from fem.finiteelement import LinearFiniteElement2d, VectorFiniteElement2d
from fem.functionspace import FunctionSpace
from fem.function import Function
from fem.auxilliary import save_to_vtk
from fem.algorithms import interpolate


def f(x):
    return np.sin(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])


element = LinearFiniteElement2d()
mesh = RectangleMesh(nref=5)
fs = FunctionSpace(mesh, element)

u = Function(fs, "u")
interpolate(f, u)

save_to_vtk(u, "u.vtk")
