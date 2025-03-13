import numpy as np


def save_to_vtk(u, filename):
    """Save a piecewise linear function as a vtk file

    :arg u: function to save
    :arg filename: name of file to save to
    """
    mesh = u.functionspace.mesh
    assert u.ndof == mesh.nvertices
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("function", file=f)
        print("ASCII", file=f)
        print("DATASET UNSTRUCTURED_GRID", file=f)
        print(f"POINTS {mesh.nvertices} float", file=f)
        for j in range(mesh.nvertices):
            print(
                f"{mesh.vertices[j,0]} {mesh.vertices[j,1]} {mesh.vertices[j,0]}",
                file=f,
            )
        print(f"CELLS {mesh.nvertices} 3", file=f)
        for cell in range(mesh.ncells):
            vertices = [
                mesh.facet2vertex[facet][0]
                for facet in [mesh.cell2facet[cell][(j - 1) % 3] for j in range(3)]
            ]
            print(f"3 {vertices[0]} {vertices[1]} {vertices[2]}", file=f)
        print(f"CELL_TYPES {mesh.nvertices}", file=f)
        for cell in range(mesh.ncells):
            print("5", file=f)
        print(f"POINT_DATA {mesh.nvertices}", file=f)
        print(f"SCALARS sample_scalars float 1", file=f)
        print(f"LOOKUP_TABLE default", file=f)
        for j in range(u.ndof):
            print(u.data[j], file=f)
