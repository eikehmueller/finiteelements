"""Main program"""

import sys
import numpy as np

import petsc4py

petsc4py.init("-ksp_type preonly -pc_type lu")
from petsc4py import PETSc

from fem.utilitymeshes import RectangleMesh
from fem.polynomialelement import PolynomialElement
from fem.functionspace import FunctionSpace
from fem.function import Function, CoFunction
from fem.algorithms import (
    interpolate,
    assemble_rhs,
    assemble_lhs,
    assemble_lhs_sparse,
    two_norm,
)
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle


def f(x):
    """function to interpolate"""
    return np.cos(2 * np.pi * x[0]) * np.cos(4 * np.pi * x[1])


def test_solve():
    """Check that dense matrix assembly and solve works"""
    # Polynomial degree
    degree = 3
    nref = 4
    # Coeffcient of diffusion term
    kappa = 0.9
    # Coefficient of zero order term
    omega = 0.4

    element = PolynomialElement(degree)

    mesh = RectangleMesh(Lx=1, Ly=1, nref=nref)
    fs = FunctionSpace(mesh, element)

    quad = GaussLegendreQuadratureReferenceTriangle(degree + 1)

    r = CoFunction(fs)
    assemble_rhs(f, r, quad)
    r.data[:] *= (2**2 + 4**2) * np.pi**2 * kappa + omega

    u_numerical = Function(fs, "u_numerical")

    stiffness_matrix = assemble_lhs(fs, quad, kappa, omega)
    u_numerical.data[:] = np.linalg.solve(stiffness_matrix, r.data[:])

    u_exact = Function(fs, "u_exact")

    interpolate(f, u_exact)

    error = Function(fs, "error")
    error.data[:] = u_numerical.data[:] - u_exact.data[:]

    error_norm = two_norm(error, quad)

    tolerance = 1.0e-4
    assert error_norm < tolerance


def test_solve_sparse():
    """Check that sparse matrix assembly and solve works"""
    # Polynomial degree
    degree = 3
    nref = 5
    # Coeffcient of diffusion term
    kappa = 0.9
    # Coefficient of zero order term
    omega = 0.4

    element = PolynomialElement(degree)

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

    u_exact = Function(fs, "u_exact")

    interpolate(f, u_exact)

    error = Function(fs, "error")
    error.data[:] = u_numerical.data[:] - u_exact.data[:]

    error_norm = two_norm(error, quad)

    tolerance = 1.0e-4
    assert error_norm < tolerance
