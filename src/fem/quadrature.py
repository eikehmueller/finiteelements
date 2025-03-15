"""Quadrature rules on reference triangle"""

from abc import ABC, abstractmethod
import numpy as np

__all__ = ["Quadrature", "GaussLegendreQuadrature"]


class Quadrature(ABC):
    """Base class for quadrature in 2d"""

    @property
    @abstractmethod
    def nodes(self):
        """Return quadrature nodes"""

    @property
    @abstractmethod
    def weights(self):
        """Return quadrature weights"""


class GaussLegendreQuadrature(Quadrature):
    """Gauss-Legendre quadrature on reference triangle

    The tensor-product of one-dimensional Gauss-Legendre quadrature of given degree is
    mapped from [-1,+1] x [-1,+1] to the reference triangle with the Duffy transform.
    Note that for a given degree p, polynomials of degree 2*p-1 or less are integrated
    exactly.
    """

    def __init__(self, degree):
        """Initialise a new instance

        :arg degree: polynomial degree
        """
        super().__init__()
        assert degree >= 1
        self._degree = degree
        xi1d, weights1d = np.polynomial.legendre.leggauss(degree)
        xi = []
        weights = []
        for x, w_x in zip(xi1d, weights1d):
            for y, w_y in zip(xi1d, weights1d):
                x_sq = 1 / 2 * (1 + x)
                y_sq = 1 / 2 * (1 + y)
                xi.append([x_sq, (1 - x_sq) * y_sq])
                weights.append(1 / 4 * w_x * w_y * (1 - x_sq))
        self._nodes = np.asarray(xi)
        self._weights = np.asarray(weights)

    @property
    def nodes(self):
        """Return quadrature nodes"""
        return self._nodes

    @property
    def weights(self):
        """Return quadrature weights"""
        return self._weights
