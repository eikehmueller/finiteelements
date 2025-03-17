"""Main program"""

import numpy as np

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.utilities import save_to_vtk, visualise_mesh, visualise_element
from fem.algorithms import interpolate, assemble_rhs, assemble_lhs
from fem.quadrature import GaussLegendreQuadrature


def f(x):
    """function to interpolate"""
    return np.sin(2 * np.pi * x[0]) * np.sin(4 * np.pi * x[1])


element = LinearElement()
mesh = RectangleMesh(Lx=1, Ly=1, nref=4)
fs = FunctionSpace(mesh, element)
u = Function(fs, "u")
interpolate(f, u)

save_to_vtk(u, "u.vtk")
# visualise_mesh(mesh, "mesh.pdf")


element = PolynomialElement(2)
visualise_element(element, "element.pdf")

quad = GaussLegendreQuadrature(3)
r = CoFunction(fs)
assemble_rhs(f, r, quad)
xi = np.asarray([0.4, 0.3])

stiffness_matrix = assemble_lhs(fs, quad)

from matplotlib import pyplot as plt

plt.clf()
plt.spy(stiffness_matrix)
plt.savefig("stiffness_matrix.pdf", bbox_inches="tight")
