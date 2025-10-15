"""Some utilities for visualisation etc."""

from contextlib import contextmanager
import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from fem.linearelement import LinearElement

__all__ = ["measure_time", "save_to_vtk", "plot_solution", "grid_function"]


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

    valid_element = isinstance(element, LinearElement)
    try:
        from fem.polynomialelement import PolynomialElement

        valid_element = valid_element or (
            isinstance(element, PolynomialElement) and (element.degree == 1)
        )
    except:
        pass

    assert valid_element, "Only linear finite elements can be saved in vtk format"
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
    Z_exact = u_exact(XY.T).reshape([X.shape[0], X.shape[1]]).T
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
    t_new = np.arange(int(np.ceil(t_old[0])), int(np.floor(t_old[-1])) + 1)
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
    plt.savefig(filename, bbox_inches="tight", dpi=300)


def grid_function(u, nx=256, ny=256):
    """Convert function to plottable arrays

    The smallest rectangle that covers the domain is sub-divided into a grid with nx cells in
    the horizontal (x-) direction and ny cells in the vertical (y-) direction. The function
    u(x) is evaluated at all (nx+1)*(ny+1) vertices of the resulting grid.

    Three arrays of shape (nx+1,ny+1) are returned:
        * X: the x-coordinates of all vertices of the grid
        * Y: the y-coordinates of all vertices of the grid
        * Z: the value of the function at all vertices of the grid

    It is implicitly assumed that the coordinates of the mesh are represented by piecewise
    linear functions. The code will still work if this is not the case, but the grid might be
    distorted.

    :arg u: function to grid
    :arg nx: number of grid-cells in the horizontal (x-) direction
    :arg ny: number of grid-cells in the vertical (y-) direction
    """
    # Number of cells used for discretisation
    n = np.asarray([nx, ny])
    mesh = u.functionspace.mesh
    if not (
        type(mesh.coordinates.functionspace.finiteelement.scalarfiniteelement)
        is LinearElement
    ):
        print("WARNING: Plot might be distorted")

    fs_coord = mesh.coordinates.functionspace
    element_coord = fs_coord.finiteelement
    zeta = np.asarray([[0, 0], [1, 0], [0, 1]])
    coord_table = np.transpose(element_coord.tabulate(zeta), (0, 2, 1))
    # Loop over all cells of mesh and find largest and smallest coordinates
    p_max = np.asarray([[-np.inf, -np.inf]])
    p_min = np.asarray([[np.inf, np.inf]])
    for alpha in range(mesh.ncells):
        # local dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        # global dof-indices of coordinate field
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # coordinates
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        vertex_coordinates = coord_table @ x_dof_vector
        p_max = np.max(
            np.concatenate((vertex_coordinates, p_max)), axis=0, keepdims=True
        )
        p_min = np.min(
            np.concatenate((vertex_coordinates, p_min)), axis=0, keepdims=True
        )
    # Convert coordinates to array of shape (n,2)
    offset = p_min.flatten()
    h = (p_max.flatten() - offset) / n

    XY = [
        np.arange(offset[j], offset[j] + (n[j] + 1 / 2) * h[j], h[j]) for j in range(2)
    ]
    mg = np.asarray(np.meshgrid(*XY)).T
    fs = u.functionspace
    element = fs.finiteelement
    Z = np.zeros(mg.shape[:-1])
    indices = np.asarray(
        np.meshgrid(*[np.arange(0, n[j] + 1) for j in range(2)]), dtype=int
    ).T
    for alpha in range(mesh.ncells):
        # local dof-indices of coordinate field
        ell_coord = range(element_coord.ndof)
        # global dof-indices of coordinate field
        ell_g_coord = fs_coord.local2global(alpha, ell_coord)
        # coordinates
        x_dof_vector = mesh.coordinates.data[ell_g_coord]
        vertex_coordinates = coord_table @ x_dof_vector
        A_inv = np.linalg.inv(
            np.asarray(
                [
                    vertex_coordinates[1, :] - vertex_coordinates[0, :],
                    vertex_coordinates[2, :] - vertex_coordinates[0, :],
                ]
            )
        )
        # local coordinates
        w = (mg - vertex_coordinates[0, :]) @ A_inv
        filter = np.logical_and(
            np.logical_and(w[..., 0] >= 0, w[..., 1] >= 0), w[..., 0] + w[..., 1] <= 1
        )
        # local dof-indices of dof-vector
        ell = range(element.ndof)
        ell_g = fs.local2global(alpha, ell)
        dof_vector = u.data[ell_g]
        zeta = w[filter]
        values = element.tabulate(zeta) @ dof_vector
        # Set values
        Z[*indices[filter].T] = values
    return mg[..., 0], mg[..., 1], Z
