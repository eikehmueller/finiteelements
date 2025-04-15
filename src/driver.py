"""Main program"""

import sys
import numpy as np

import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.utilities import save_to_vtk
from fem.algorithms import interpolate, assemble_rhs, assemble_lhs, two_norm
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle


def f(x):
    """function to interpolate"""
    return np.cos(2 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])


nref = 6

element = LinearElement()
mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
fs = FunctionSpace(mesh, element)

quad = GaussLegendreQuadratureReferenceTriangle(3)

r = CoFunction(fs)
assemble_rhs(f, r, quad)
r.data[:] *= 20 * np.pi**2 + 1

u_numerical = Function(fs, "u_numerical")

stiffness_matrix = assemble_lhs(fs, quad, sparse=True)
u_petsc = PETSc.Vec().createWithArray(u_numerical.data)
r_petsc = PETSc.Vec().createWithArray(r.data)

ksp = PETSc.KSP().create()
ksp.setOperators(stiffness_matrix)
ksp.setFromOptions()
ksp.solve(r_petsc, u_petsc)

u_exact = Function(fs, "u_exact")

interpolate(f, u_exact)

error = Function(fs, "error")
error.data[:] = u_numerical.data[:] - u_exact.data[:]

error_norm = two_norm(error, quad)

print(f"nref = {nref}, error = {error_norm}")

save_to_vtk(u_exact, "u_exact.vtk")
save_to_vtk(u_numerical, "u_numerical.vtk")
save_to_vtk(error, "error.vtk")
