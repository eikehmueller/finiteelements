from fem.linearelement import LinearElement
from fem.polynomialelement import PolynomialElement

import numpy as np

__all__ = ["jacobian", "save_to_vtk"]


def jacobian(self, mesh, cell, xi):
    """Calculate Jacobian in a given cell for quadrature points in reference triangle

    :arg mesh: underlying mesh
    :arg cell: index of mesh cell
    :arg xi: point in reference cell at which the Jacobian is to be evaluated
    """
    fs = mesh.coordinates.functionspace
    element = fs.finiteelement
    gradient = element.tabulate_gradient(xi)
    j_g = fs.local2global(cell, range(element.ndof))
    return np.dot(self.coordinates.data[j_g], gradient)


def save_to_vtk(u, filename):
    """Save a piecewise linear function as a vtk file

    :arg u: function to save
    :arg filename: name of file to save to
    """
    element = u.functionspace.finiteelement
    assert (type(element) is LinearElement) or (
        (type(element) is PolynomialElement) and (element.degree == 1)
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
            vertices = [
                mesh.facet2vertex[facet][0]
                for facet in [mesh.cell2facet[cell][(j - 1) % 3] for j in range(3)]
            ]
            print(f"3 {vertices[0]} {vertices[1]} {vertices[2]}", file=f)
        print(file=f)
        print(f"CELL_TYPES {mesh.ncells}", file=f)
        for cell in range(mesh.ncells):
            print("5", file=f)
        print(file=f)
        print(f"POINT_DATA {mesh.nvertices}", file=f)
        label = u.label.replace(" ", "_")
        print(f"SCALARS {label} float 1", file=f)
        print(f"LOOKUP_TABLE default", file=f)
        for j in range(u.ndof):
            print(u.data[j], file=f)
