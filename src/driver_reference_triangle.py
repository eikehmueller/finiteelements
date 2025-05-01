"""Main program"""

import functools
import numpy as np


from fem.polynomialelement import PolynomialElement
from fem.algorithms_reference_triangle import (
    assemble_rhs,
    assemble_lhs,
    two_norm,
)


def u_exact(x, sigma, x0):
    """Analytical solution

    :arg x: point at which the function is evaluated
    :arg sigma: width of peak
    :arg x0: location of peak"""
    return np.exp(
        -1 / (2 * sigma**2) * ((x[..., 0] - x0[0]) ** 2 + (x[..., 1] - x0[1]) ** 2)
    )


def f(x, kappa, omega, sigma, x0):
    """function to interpolate

    :arg x: point at which the function is evaluated
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero-order term
    :arg sigma: width of peak
    :arg x0: location of peak
    """
    x_sq = (x[..., 0] - x0[0]) ** 2 + (x[..., 1] - x0[1]) ** 2
    return (2 * kappa / sigma**2 + omega - kappa / sigma**4 * x_sq) * u_exact(
        x, sigma, x0
    )


def g(x, kappa, sigma, x0):
    """boundary function

    :arg x: point at which the function is evaluated
    :arg kappa: coefficient of diffusion term
    :arg sigma: width of peak
    :arg x0: location of peak
    """
    if np.all(x[..., 1]) < 1e-12:
        # facet F_1
        n_dot_x = -(x[..., 1] - x0[1])
    elif np.all(x[..., 0]) < 1e-12:
        # facet F_2
        n_dot_x = -(x[..., 0] - x0[0])
    else:
        # facet F_0
        n_dot_x = (x[..., 0] - x0[0] + x[..., 1] - x0[1]) / np.sqrt(2)
    return -kappa / sigma**2 * n_dot_x * u_exact(x, sigma, x0)


def plot_solution(u_numerical, sigma, x0, filename):
    """Plot numerical solution, exact solution and error

    :arg u_numerical: numerical solution
    :arg filename: name of file to save plot to
    """
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    h = 0.01
    X = np.arange(0, 1 + h / 2, h)
    Y = np.arange(0, 1 + h / 2, h)
    X, Y = np.meshgrid(X, Y)

    XY = np.asarray([X, Y]).T.reshape([X.shape[0] * X.shape[1], 2])

    T = element.tabulate(XY)
    Z_exact = u_exact(XY, sigma, x0).reshape([X.shape[0], X.shape[1]]).T
    Z_numerical = np.dot(T, u_numerical).reshape([X.shape[0], X.shape[1]]).T

    fig, axs = plt.subplots(1, 3)
    cs = axs[0].contourf(X, Y, Z_exact, levels=100, vmin=0, vmax=1)
    cbar = fig.colorbar(cs, shrink=1, location="bottom")
    cbar.ax.tick_params(labelsize=4)
    axs[0].set_title("exact")

    cs = axs[2].contourf(X, Y, Z_numerical, levels=100, vmin=0, vmax=1)
    cbar = fig.colorbar(cs, shrink=1, location="bottom")
    cbar.ax.tick_params(labelsize=4)
    axs[2].set_title("numerical")

    cs = axs[1].contourf(
        X, Y, np.log(np.abs(Z_numerical - Z_exact)), levels=100, norm="linear"
    )
    cbar = fig.colorbar(cs, shrink=1, location="bottom")
    cbar.ax.tick_params(labelsize=4)
    axs[1].set_title("log(error)")

    for ax in axs:
        ax.set_aspect("equal")
        border = 2
        mask = Polygon(
            [[1 + border, -border], [-border, 1 + border], [1 + border, 1 + border]],
            color="white",
            linewidth=4,
        )
        p = PatchCollection([mask], color="white", zorder=2)
        ax.add_collection(p)
    plt.savefig(filename, bbox_inches="tight")


# width of peak
sigma = 0.5
# location of peak
x0 = np.asarray([0.6, 0.25])
# Coefficient of diffusion term
kappa = 0.9
# Coefficient of zero order term
omega = 0.4
# Polynomial degree
degree = 4
# Quadrature parameter
n_q = degree + 1

element = PolynomialElement(degree)

stiffness_matrix = assemble_lhs(element, n_q, kappa, omega)
r = assemble_rhs(
    functools.partial(f, kappa=kappa, omega=omega, sigma=sigma, x0=x0),
    functools.partial(g, kappa=kappa, sigma=sigma, x0=x0),
    element,
    n_q,
)

u_numerical = np.linalg.solve(stiffness_matrix, r)
u = element.tabulate_dofs(functools.partial(u_exact, sigma=sigma, x0=x0))
error = u_numerical - u

rel_error_nrm = two_norm(error, element, n_q) / two_norm(u, element, n_q)
condition_number = np.linalg.cond(stiffness_matrix)
print(f"{degree}: {rel_error_nrm**2:10.4e},")

plot_solution(u_numerical, sigma, x0, "triangle_solution.pdf")
