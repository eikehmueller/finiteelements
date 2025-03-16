"""Main program"""

import numpy as np

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.functionspace import FunctionSpace
from fem.function import Function
from fem.utilities import save_to_vtk, visualise_mesh
from fem.algorithms import interpolate


def f(x):
    """function to interpolate"""
    return np.sin(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])


element = LinearElement()
mesh = RectangleMesh(Lx=1, Ly=1, nref=1)
fs = FunctionSpace(mesh, element)

u = Function(fs, "u")
interpolate(f, u)

save_to_vtk(u, "u.vtk")
visualise_mesh(mesh, "mesh.pdf")
