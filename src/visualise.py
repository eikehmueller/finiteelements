"""Visualise mesh, finite element and quadrature"""

import numpy as np

from fem.utilitymeshes import RectangleMesh

from fem.utilities import (
    visualise_mesh,
    visualise_element,
    visualise_quadrature,
)
from fem.quadrature import GaussLegendreQuadratureReferenceTriangle


def to_latex(matrix):
    """Convert a numpy matrix to a latex string

    :arg matrix: Matrix to convert
    """
    _matrix = np.asarray(matrix)
    nrow, ncol = _matrix.shape
    s = ""
    s += r"\begin{pmatrix}" + "\n"
    for j in range(nrow):
        s += " & ".join([str(x) for x in _matrix[j, :]])
        if j < nrow - 1:
            s += r"\\"
        s += "\n"
    s += r"\end{pmatrix}"
    return s


mesh = RectangleMesh(Lx=1.0, Ly=1.0, nref=1)

print("cell2facet:")
print(to_latex(np.asarray(mesh.cell2facet).T))
print()
print("facet2vertex:")
print(to_latex(np.asarray(mesh.facet2vertex).T))
print()
print("cell2vertex:")
print(to_latex(np.asarray(mesh.cell2vertex).T))
print()

visualise_mesh(mesh, "mesh.svg")

try:
    from fem.polynomialelement import PolynomialElement

    element = PolynomialElement(2)
    visualise_element(element, "element.png")
except:
    pass

quad = GaussLegendreQuadratureReferenceTriangle(3)
visualise_quadrature(quad, "quadrature.pdf")
