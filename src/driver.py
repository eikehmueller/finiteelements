"""Solve finite element model proble,m"""

import numpy as np
import functools
import sys

import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.utilities import save_to_vtk, measure_time
from fem.algorithms import assemble_rhs, assemble_lhs, assemble_lhs_sparse, error_nrm
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle


def f(x, s):
    """function to project

    :arg x: point at which to evaluate the function
    :arg s: wave index
    """
    return np.cos(np.pi * s[0] * x[0]) * np.cos(np.pi * s[1] * x[1])


# Number of mesh refinements
nref = 5
# Coeffcient of diffusion term
kappa = 0.9
# Coefficient of zero order term
omega = 0.4
# Wave vector
s = [2, 4]
# Polynomial degree of the finite element
degree = 1

# Finite element
if degree == 1:
    element = LinearElement()
else:
    element = PolynomialElement(degree)
# Mesh
mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
# Function space
fs = FunctionSpace(mesh, element)
# Quadrature rule
quad = GaussLegendreQuadratureReferenceTriangle(2)
# Use sparse matrices
sparse_matrices = True

print(f"nref = {nref}")
print(f"number of unknowns = {fs.ndof}")
print()

# Construct right hand side
# (s_0^2 + s_1^2)*pi^2*u(x)
b_h = CoFunction(fs)
with measure_time("assemble right hand side"):
    assemble_rhs(functools.partial(f, s=s), b_h, quad)
b_h.data[:] *= (s[0] ** 2 + s[1] ** 2) * np.pi**2 * kappa + omega

# Numerical solution
u_h = Function(fs, "u_numerical")

# Stiffness matrix
with measure_time("assemble stiffness matrix"):
    if sparse_matrices:
        stiffness_matrix = assemble_lhs_sparse(fs, quad, kappa, omega)
    else:
        stiffness_matrix = assemble_lhs(fs, quad, kappa, omega)

# Solve linear system M^{(h)} u^{(h)} = b^{(h)}
with measure_time("solve linear system"):
    if sparse_matrices:
        u_petsc = PETSc.Vec().createWithArray(u_h.data)
        b_petsc = PETSc.Vec().createWithArray(b_h.data)
        ksp = PETSc.KSP().create()
        ksp.setOperators(stiffness_matrix)
        ksp.setFromOptions()
        ksp.solve(b_petsc, u_petsc)
        niter = ksp.getIterationNumber()
    else:
        u_h.data[:] = np.linalg.solve(stiffness_matrix, b_h.data)

u_exact = Function(fs, "u_exact")

b_h = CoFunction(fs)
assemble_rhs(functools.partial(f, s=s), b_h, quad)

# Solve linear system M^{(h)} u_{exact} = b^{(h)}
if sparse_matrices:
    mass_matrix = assemble_lhs_sparse(fs, quad, 0, 1)
    u_petsc = PETSc.Vec().createWithArray(u_exact.data)
    b_petsc = PETSc.Vec().createWithArray(b_h.data)
    ksp = PETSc.KSP().create()
    ksp.setOperators(mass_matrix)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-12)
    ksp.solve(b_petsc, u_petsc)
else:
    mass_matrix = assemble_lhs(fs, quad, 0, 1)
    u_exact.data[:] = np.linalg.solve(mass_matrix, b_h.data)

error_norm = error_nrm(u_h, functools.partial(f, s=s), quad)

if sparse_matrices:
    print()
    n_nz = int(stiffness_matrix.getInfo()["nz_used"])
    print(f"number of non-zero matrix entries = {n_nz}")
    print(f"number of solver iterations = {niter}")

print()
print(f"error = {error_norm}")

# Compute error
error = Function(fs, "error")
error.data[:] = u_h.data[:] - u_exact.data[:]

# Save solution and error to .vtk file
if element is LinearElement:
    save_to_vtk(u_exact, "u_exact.vtk")
    save_to_vtk(u_h, "u_numerical.vtk")
    save_to_vtk(error, "error.vtk")
