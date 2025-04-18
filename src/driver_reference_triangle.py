"""Main program"""

import functools
import numpy as np


from fem.polynomialelement import PolynomialElement
from fem.algorithms_reference_triangle import (
    assemble_rhs,
    assemble_lhs,
    two_norm,
)


def f(x, kappa, omega):
    """function to interpolate

    :arg x: point at which the function is evaluated
    :arg kappa: coefficient of diffusion term
    :arg omega: coefficient of zero-order term
    """
    return (
        (omega + 2 * np.pi**2 * kappa)
        * np.sin(np.pi * x[..., 0])
        * np.sin(np.pi * x[..., 1])
    )


def g(x, kappa):
    """boundary function

    :arg x: point at which the function is evaluated
    :arg kappa: coefficient of diffusion term
    """
    if np.all(x[..., 1]) < 1e-12:
        return -np.pi * kappa * np.sin(np.pi * x[..., 0])
    if np.all(x[..., 0]) < 1e-12:
        return -np.pi * kappa * np.sin(np.pi * x[..., 1])
    else:
        return kappa * (
            np.pi
            / np.sqrt(2)
            * (
                np.cos(np.pi * x[..., 0]) * np.sin(np.pi * x[..., 1])
                + np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1])
            )
        )


def plot_solution(u_numerical, filename):
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

    Z_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

    XY = np.asarray([X, Y]).T.reshape([X.shape[0] * X.shape[1], 2])

    T = element.tabulate(XY)
    Z_numerical = np.dot(T, u_numerical).reshape([X.shape[0], X.shape[1]])

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
        sigma = 2
        mask = Polygon(
            [[1 + sigma, -sigma], [-sigma, 1 + sigma], [1 + sigma, 1 + sigma]],
            color="white",
            linewidth=4,
        )
        p = PatchCollection([mask], color="white", zorder=2)
        ax.add_collection(p)
    plt.savefig(filename, bbox_inches="tight")


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
    functools.partial(f, kappa=kappa, omega=omega),
    functools.partial(g, kappa=kappa),
    element,
    n_q,
)

u_numerical = np.linalg.solve(stiffness_matrix, r)
u_exact = element.tabulate_dofs(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
error = u_numerical - u_exact

error_nrm = two_norm(error, element, n_q)
print(f"degree = {degree}")
condition_number = np.linalg.cond(stiffness_matrix)
print(f"condition number = {condition_number:8.2e}")
print(f"error norm:      {error_nrm:8.3e}")

plot_solution(u_numerical, "triangle_solution.pdf")
