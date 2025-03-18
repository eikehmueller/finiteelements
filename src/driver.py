"""Main program"""

import numpy as np

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.utilities import save_to_vtk
from fem.algorithms import interpolate, assemble_rhs, assemble_lhs
from fem.quadrature import GaussLegendreQuadrature


def f(x):
    """function to interpolate"""
    return np.cos(2 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])


element = LinearElement()
mesh = RectangleMesh(Lx=1, Ly=1, nref=6)
fs = FunctionSpace(mesh, element)

element = PolynomialElement(2)
quad = GaussLegendreQuadrature(3)

r = CoFunction(fs)
assemble_rhs(f, r, quad)
r.data[:] *= 20 * np.pi**2 + 1

u_numerical = Function(fs, "u_numerical")

stiffness_matrix = assemble_lhs(fs, quad)

u_numerical.data[:] = np.linalg.solve(stiffness_matrix, r.data[:])

u_exact = Function(fs, "u_exact")

interpolate(f, u_exact)

error = Function(fs, "error")
error.data[:] = u_numerical.data[:] - u_exact.data[:]

save_to_vtk(u_exact, "u_exact.vtk")
save_to_vtk(u_numerical, "u_numerical.vtk")
save_to_vtk(error, "error.vtk")
