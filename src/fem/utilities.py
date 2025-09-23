"""Some utilities for visualisation etc."""

from contextlib import contextmanager
import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement

__all__ = [
    "measure_time",
    "save_to_vtk",
    "visualise_mesh",
    "visualise_element",
    "visualise_quadrature",
]


@contextmanager
def measure_time(label):
    """Measure the time it takes to execute a block of code

    :arg label: label for the time measurement
    """
    t_start = time.perf_counter()
    try:
        yield
    finally:
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"time [{label}] = {t_elapsed:8.2e} s")


def save_to_vtk(u, filename):
    """Save a piecewise linear function as a vtk file

    :arg u: function to save
    :arg filename: name of file to save to
    """
    element = u.functionspace.finiteelement
    assert isinstance(element, LinearElement) or (
        isinstance(element, PolynomialElement) and (element.degree == 1)
    ), "Only linear finite elements can be saved in vtk format"
    mesh = u.functionspace.mesh
    assert u.ndof == mesh.nvertices
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("function", file=f)
        print("ASCII", file=f)
        print("DATASET UNSTRUCTURED_GRID", file=f)
        print(file=f)
        print(f"POINTS {mesh.nvertices} float", file=f)
        for j in range(mesh.nvertices):
            print(
                f"{mesh.vertices[j,0]} {mesh.vertices[j,1]} 0",
                file=f,
            )
        print(file=f)
        print(f"CELLS {mesh.ncells} {4*mesh.ncells}", file=f)
        for cell in range(mesh.ncells):
            vertices = mesh.cell2vertex[cell]
            print(f"3 {vertices[0]} {vertices[1]} {vertices[2]}", file=f)
        print(file=f)
        print(f"CELL_TYPES {mesh.ncells}", file=f)
        for cell in range(mesh.ncells):
            print("5", file=f)
        print(file=f)
        print(f"POINT_DATA {mesh.nvertices}", file=f)
        label = u.label.replace(" ", "_")
        print(f"SCALARS {label} float 1", file=f)
        print("LOOKUP_TABLE default", file=f)
        for j in range(u.ndof):
            print(u.data[j], file=f)


def visualise_mesh(mesh, filename):
    """Plot connectivity information for a mesh

    :arg mesh: mesh object
    :arg filename: name of file to save output to
    """
    plt.clf()
    _, axs = plt.subplots(2, 2)
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.25, wspace=0.4)
    for j in range(2):
        for k in range(2):
            axs[j, k].set_aspect("equal")
    # vertices
    for cell in range(mesh.ncells):
        p = np.zeros((4, 3))
        for j in range(3):
            p[j, :2] = np.asarray(mesh.vertices[mesh.cell2vertex[cell][j]])
        p[-1, :] = p[0, :]
        axs[0, 0].plot(
            p[:, 0],
            p[:, 1],
            linewidth=2,
            color="lightgray",
        )
    axs[0, 0].plot(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        linewidth=0,
        marker="o",
        markersize=4,
        color="blue",
    )
    for vertex in range(mesh.nvertices):
        axs[0, 0].annotate(
            f"{vertex:3d}",
            (mesh.vertices[vertex, :]),
            fontsize=6,
        )
        axs[0, 0].set_title("global vertex index")
    # facets
    for facet in range(mesh.nfacets):
        p = np.asarray([mesh.vertices[vertex] for vertex in mesh.facet2vertex[facet]])
        m = np.mean(p, axis=0)
        rho = 0.8
        p = rho * p + (1 - rho) * np.expand_dims(m, axis=0)
        axs[0, 1].plot(
            p[:, 0], p[:, 1], linewidth=2, marker="o", markersize=4, color="blue"
        )
        axs[0, 1].annotate(
            f"{facet:3d}",
            (m[0], m[1]),
            fontsize=6,
        )
        axs[0, 1].arrow(
            p[0, 0],
            p[0, 1],
            m[0] - p[0, 0],
            m[1] - p[0, 1],
            width=0,
            linewidth=0,
            head_width=0.05,
            color="blue",
        )
    axs[0, 1].set_title("global facet index")
    # cells
    for cell in range(mesh.ncells):
        p = np.zeros((4, 3))
        for j in range(3):
            p[j, :2] = np.asarray(mesh.vertices[mesh.cell2vertex[cell][j]])
        p[-1, :] = p[0, :]
        m = np.mean(p[:-1, :], axis=0)
        rho = 0.8
        p = rho * p + (1 - rho) * np.expand_dims(m, axis=0)
        axs[1, 0].plot(p[:, 0], p[:, 1], linewidth=2, color="blue")
        axs[1, 0].annotate(
            f"{cell:3d}",
            (m[0], m[1]),
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=6,
        )
    axs[1, 0].set_title("global cell index")
    # local indices
    for cell in range(mesh.ncells):
        p = np.zeros((4, 3))
        p_facet = np.zeros((3, 2))
        for j, facet in enumerate(mesh.cell2facet[cell]):
            p_facet[j, :] = 0.5 * (
                mesh.vertices[mesh.facet2vertex[facet][0]]
                + mesh.vertices[mesh.facet2vertex[facet][1]]
            )
            p[j, :2] = np.asarray(mesh.vertices[mesh.cell2vertex[cell][j]])
        p[-1, :] = p[0, :]
        m = np.mean(p[:-1, :], axis=0)
        rho = 0.85
        p_cell = rho * p + (1 - rho) * np.expand_dims(m, axis=0)
        axs[1, 1].plot(
            p_cell[:, 0],
            p_cell[:, 1],
            linewidth=2,
            color="black",
            markersize=4,
            marker="o",
            markerfacecolor="red",
        )
        axs[1, 1].annotate(
            f"{cell:3d}",
            (m[0], m[1]),
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=6,
        )
        omega = 0.6
        p_vertex = omega * p + (1 - omega) * np.expand_dims(m, axis=0)
        p_facet = omega * p_facet + (1 - omega) * np.expand_dims(m[:2], axis=0)
        for j in range(3):
            axs[1, 1].annotate(
                f"{j:d}",
                (p_vertex[j, 0], p_vertex[j, 1]),
                verticalalignment="center",
                horizontalalignment="center",
                color="red",
                fontsize=6,
            )
            axs[1, 1].annotate(
                f"{j:d}",
                p_facet[j, :],
                verticalalignment="center",
                horizontalalignment="center",
                color="blue",
                fontsize=6,
            )
            axs[1, 1].arrow(
                p_cell[j, 0],
                p_cell[j, 1],
                0.5 * (p_cell[(j + 1) % 3, 0] - p_cell[j, 0]),
                0.5 * (p_cell[(j + 1) % 3, 1] - p_cell[j, 1]),
                width=0,
                linewidth=1,
                head_width=0.025,
                color="black",
            )
    axs[1, 1].set_title("local vertex- and facet-index")
    plt.savefig(filename, bbox_inches="tight")


def balanced_factorisation(n):
    """factorise n = a*b such that a+b is minimal and a<b

    :arg n: number to factorise
    """
    s = dict()
    for a in range(1, n + 1):
        if (n // a) * a == n:
            s[(a, n // a)] = a + n // a
    return min(s, key=s.get)


def visualise_element(element, filename):
    """Visualise the basis functions of a finite element

    :arg element: finite element
    :arg filename: name of file that the visualisation is saved to
    """

    plt.clf()
    ndof = element.ndof
    nrows, ncols = balanced_factorisation(ndof)
    fig, axs = plt.subplots(nrows, ncols)
    h = 0.01
    X = np.arange(0, 1 + h / 2, h)
    Y = np.arange(0, 1 + h / 2, h)
    X, Y = np.meshgrid(X, Y)

    xi = np.asarray((X.flatten(), Y.flatten())).T
    Z = element.tabulate(xi).reshape((*X.shape, ndof))

    h_c = 0.1
    X_c = np.arange(0, 1 + h_c / 2, h_c)
    Y_c = np.arange(0, 1 + h_c / 2, h_c)
    X_c, Y_c = np.meshgrid(X_c, Y_c)
    xi_c = np.asarray((X_c.flatten(), Y_c.flatten())).T
    gradZ = element.tabulate_gradient(xi_c).reshape((*X_c.shape, ndof, 2))
    for j in range(ndof):
        row = j // ncols
        col = j % ncols
        if nrows == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]
        if col > 0:
            ax.set_yticks([])
        if row < nrows - 1:
            ax.set_xticks([])
        ax.set_aspect(1)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(j)
        ax.contourf(X, Y, Z[..., j], levels=100, vmin=0, vmax=1)
        ax.quiver(X_c, Y_c, gradZ[..., j, 0], gradZ[..., j, 1], color="red")
        sigma = 2
        mask = Polygon(
            [[1 + sigma, -sigma], [-sigma, 1 + sigma], [1 + sigma, 1 + sigma]],
            color="white",
            linewidth=4,
        )
        p = PatchCollection([mask], color="white", zorder=2)
        ax.add_collection(p)

        ax.plot(
            element._nodal_points[:, 0],
            element._nodal_points[:, 1],
            linewidth=0,
            markersize=4,
            markerfacecolor="white",
            marker="o",
            color="red",
        )
        ax.plot(
            element._nodal_points[j, 0],
            element._nodal_points[j, 1],
            linewidth=0,
            markersize=4,
            marker="o",
            color="red",
        )

    plt.savefig(filename, bbox_inches="tight")


def visualise_quadrature(quad, filename):
    """Visualise quadrature rule

    :arg quad: quadrature rule to visualise
    :arg filename: name of file to write to
    """
    plt.clf()
    mask = Polygon(
        [[0, 0], [1, 0], [0, 1]],
        color="lightgray",
        linewidth=2,
    )
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    p = PatchCollection([mask], color="lightgray")
    ax.add_collection(p)
    plt.plot(
        quad.nodes[:, 0],
        quad.nodes[:, 1],
        linewidth=0,
        marker="o",
        markersize=1,
        color="blue",
    )
    plt.savefig(filename, bbox_inches="tight")


def plot_solution(u_numerical, u_exact, element, filename):
    """Plot numerical solution, exact solution and error

    :arg u_numerical: numerical solution vector
    :arg u_exact: exact solution function
    :arg element: finite element
    :arg filename: name of file to save plot to
    """

    h = 0.01
    X = np.arange(0, 1 + h / 2, h)
    Y = np.arange(0, 1 + h / 2, h)
    X, Y = np.meshgrid(X, Y)

    XY = np.asarray([X, Y]).T.reshape([X.shape[0] * X.shape[1], 2])

    T = element.tabulate(XY)
    Z_exact = u_exact(XY).reshape([X.shape[0], X.shape[1]]).T
    Z_exact[0, -1] = 0
    Z_exact[0, -2] = 1
    Z_numerical = np.dot(T, u_numerical).reshape([X.shape[0], X.shape[1]]).T
    Z_numerical[0, -1] = 0
    Z_numerical[0, -2] = 1

    fig, axs = plt.subplots(1, 3)
    cs = axs[0].contourf(
        X,
        Y,
        np.clip(Z_exact, a_min=0, a_max=1),
        levels=100,
        norm="linear",
        vmin=0,
        vmax=1,
    )
    cbar = fig.colorbar(
        cs,
        shrink=1,
        location="bottom",
    )
    cbar.ax.tick_params(labelsize=4, rotation=20)
    axs[0].set_title("exact", fontsize=10)

    cs = axs[2].contourf(
        X, Y, np.clip(Z_numerical, a_min=0, a_max=1), levels=100, vmin=0, vmax=1
    )
    cbar = fig.colorbar(cs, shrink=1, location="bottom", extend="both")
    cbar.ax.tick_params(labelsize=4, rotation=20)
    axs[2].set_title("numerical", fontsize=10)
    difference = Z_numerical - Z_exact
    for j in range(difference.shape[0]):
        difference[-j, j:] = None
    cs = axs[1].contourf(
        X, Y, np.log10(np.abs(difference)), levels=100, norm="linear", cmap="plasma"
    )
    cbar = fig.colorbar(cs, shrink=1, location="bottom", extend="both")
    t_old = cbar.ax.get_xticks()
    print(t_old)
    t_new = np.arange(int(np.ceil(t_old[0])), int(np.floor(t_old[-1])) + 1)
    print(t_new)
    cbar.ax.set_xticks(t_new)
    cbar.ax.set_xticklabels([f"$10^{{{t}}}$" for t in t_new])
    cbar.ax.tick_params(labelsize=4, rotation=20)
    axs[1].set_title(r"$|\text{numerical}-\text{exact}|$", fontsize=10)

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
        ax.tick_params(axis="both", which="major", labelsize=6)
    plt.savefig(filename, bbox_inches="tight")
