"""Visualise mesh, finite element and quadrature"""

from fem.utilitymeshes import RectangleMesh
from fem.polynomialelement import PolynomialElement

from fem.utilities import (
    visualise_mesh,
    visualise_element,
    visualise_quadrature,
)
from fem.quadrature import GaussLegendreQuadrature

mesh = RectangleMesh(Lx=1, Ly=1, nref=1)

visualise_mesh(mesh, "mesh.pdf")

element = PolynomialElement(2)
visualise_element(element, "element.pdf")

quad = GaussLegendreQuadrature(3)
visualise_quadrature(quad, "quadrature.pdf")
