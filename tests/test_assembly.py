"""Main program"""

import numpy as np
import pytest

import petsc4py

petsc4py.init("-ksp_type preonly -pc_type lu")
from petsc4py import PETSc

from fem.utilitymeshes import RectangleMesh
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.assembly import assemble_rhs, assemble_lhs, assemble_lhs_sparse
from fem.error import error_norm
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle
from fixtures import element


def f(x):
    """function to project"""
    return np.cos(2 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_solve(degree, element):
    """Check that dense matrix assembly and solve works"""
    # Polynomial degree
    nref = 4
    # Coeffcient of diffusion term
    kappa = 0.9
    # Coefficient of zero order term
    omega = 0.4

    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadratureReferenceTriangle(degree + 1)

    r = CoFunction(fs)
    assemble_rhs(f, r, quad)
    r.data[:] *= (2**2 + 4**2) * np.pi**2 * kappa + omega

    u_numerical = Function(fs, "u_numerical")

    stiffness_matrix = assemble_lhs(fs, quad, kappa, omega)
    u_numerical.data[:] = np.linalg.solve(stiffness_matrix, r.data[:])

    error_nrm = error_norm(u_numerical, f, quad)

    tolerance = {1: 5e-2, 2: 2e-3, 3: 1.1e-4}
    assert error_nrm < tolerance[degree]


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_solve_sparse(degree, element):
    """Check that sparse matrix assembly and solve works"""
    # Polynomial degree
    nref = 5
    # Coeffcient of diffusion term
    kappa = 0.9
    # Coefficient of zero order term
    omega = 0.4

    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadratureReferenceTriangle(degree + 1)

    r = CoFunction(fs)
    assemble_rhs(f, r, quad)
    r.data[:] *= (2**2 + 4**2) * np.pi**2 * kappa + omega

    u_numerical = Function(fs, "u_numerical")

    stiffness_matrix = assemble_lhs_sparse(fs, quad, kappa, omega)
    u_petsc = PETSc.Vec().createWithArray(u_numerical.data)
    r_petsc = PETSc.Vec().createWithArray(r.data)

    ksp = PETSc.KSP().create()
    ksp.setOperators(stiffness_matrix)
    ksp.setFromOptions()
    ksp.solve(r_petsc, u_petsc)

    error_nrm = error_norm(u_numerical, f, quad)

    tolerance = {1: 1.5e-2, 2: 5.0e-4, 3: 1.2e-4}
    assert error_nrm < tolerance[degree]
