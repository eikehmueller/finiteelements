import numpy as np

from mesh import RectangleMesh
from finiteelement import LinearFiniteElement2d
from functionspace import FunctionSpace
from function import Function
from auxilliary import save_to_vtk
from algorithms import interpolate


def f(x):
    return np.sin(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])


element = LinearFiniteElement2d()
mesh = RectangleMesh(nref=5)
fs = FunctionSpace(mesh, element)

u = Function(fs)
interpolate(f, u)

save_to_vtk(u, "u.vtk")
