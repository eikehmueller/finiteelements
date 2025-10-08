"""Solve finite element model proble,m"""

import numpy as np
from matplotlib import pyplot as plt
import functools
import sys

import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc

from fem.utilitymeshes import RectangleMesh
from fem.linearelement import LinearElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.utilities import measure_time, grid_function
from fem.algorithms import assemble_rhs, assemble_lhs, assemble_lhs_sparse, error_nrm
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle


def f(x, s):
    """function to project

    :arg x: point at which to evaluate the function
    :arg s: wave index
    """
    return np.cos(np.pi * s[0] * x[0]) * np.cos(np.pi * s[1] * x[1])


# Number of mesh refinements
nref = 4
# Coeffcient of diffusion term
kappa = 0.9
# Coefficient of zero order term
omega = 0.4
# Wave vector
s = [2, 4]
# Polynomial degree of the finite element
degree = 3

# Finite element
if degree == 1:
    element = LinearElement()
else:
    try:
        from fem.polynomialelement import PolynomialElement

        element = PolynomialElement(degree)
    except:
        pass

# Mesh
mesh = RectangleMesh(Lx=1.0, Ly=1.0, nref=nref)
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

error_norm = error_nrm(u_h, functools.partial(f, s=s), quad)

if sparse_matrices:
    print()
    n_nz = int(stiffness_matrix.getInfo()["nz_used"])
    print(f"number of non-zero matrix entries = {n_nz}")
    print(f"number of solver iterations = {niter}")

print()
print(f"error norm = {error_norm}")

# Plot results
X, Y, Z = grid_function(u_h, nx=256, ny=256)
# Exact solution
Z_exact = np.cos(np.pi * s[0] * X) * np.cos(np.pi * s[1] * Y)

plt.clf()
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
for ax in axs:
    ax.set_aspect("equal")
axs[0].set_title(r"numerical $u^{(h)}(x)$")
cs = axs[0].contourf(X, Y, Z, levels=10, vmin=-1.2, vmax=1.2)
cbar = plt.colorbar(cs, shrink=1, location="bottom", extend="both")
axs[1].set_title(r"difference $u^{(h)}(x)-u_{\text{exact}}(x)$")
cs = axs[1].contourf(X, Y, Z - Z_exact, levels=10, cmap="plasma")
cbar = plt.colorbar(cs, shrink=1, location="bottom", extend="both")
axs[2].set_title(r"exact $u_{\text{exact}}(x)$")
cs = axs[2].contourf(X, Y, Z_exact, levels=10, vmin=-1.2, vmax=1.2)
cbar = plt.colorbar(cs, shrink=1, location="bottom", extend="both")
plt.savefig("solution.png", bbox_inches="tight", dpi=300)
