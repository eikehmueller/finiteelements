"""Quadrature rules on reference triangle and line segment"""

from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Quadrature",
    "GaussLegendreQuadratureLineSegment",
    "GaussLegendreQuadratureReferenceTriangle",
]


class Quadrature(ABC):
    """Base class for quadrature rule"""

    @property
    @abstractmethod
    def nodes(self):
        """Return quadrature nodes"""

    @property
    @abstractmethod
    def weights(self):
        """Return quadrature weights"""

    @property
    @abstractmethod
    def degree_of_precision(self):
        """Degree of precision, i.e. highest polynomial degree that
        can be integrated exactly"""


class GaussLegendreQuadratureLineSegment(Quadrature):
    """Gauss-Legendre quadrature on a line segment"""

    def __init__(self, v_a, v_b, npoints):
        """Initialise a new instance

        :arg v_a: starting point of line segment
        :arg v_b: end point of line segment
        :arg npoints: number of quadrature points
        """
        super().__init__()
        assert npoints >= 1
        self._npoints = npoints
        zeta, weights = np.polynomial.legendre.leggauss(npoints + 1)
        self._nodes = np.outer(
            1 / 2 * (1 - np.asarray(zeta)), np.asarray(v_a)
        ) + np.outer(1 / 2 * (1 + np.asarray(zeta)), np.asarray(v_b))
        self._weights = (
            np.linalg.norm(np.asarray(v_b) - np.asarray(v_a)) / 2 * np.asarray(weights)
        )

    @property
    def nodes(self):
        """Return quadrature nodes"""
        return self._nodes

    @property
    def weights(self):
        """Return quadrature weights"""
        return self._weights

    @property
    def degree_of_precision(self):
        """Degree of precision, i.e. highest polynomial degree that
        can be integrated exactly"""
        return 2 * self._npoints - 1


class GaussLegendreQuadratureReferenceTriangle(Quadrature):
    """Gauss-Legendre quadrature on reference triangle

    The tensor-product of one-dimensional Gauss-Legendre quadrature of given degree is
    mapped from [-1,+1] x [-1,+1] to the reference triangle with the Duffy transform.
    Note that for a given number of points n, polynomials of degree 2*n-1 or less are
    integrated exactly.
    """

    def __init__(self, npoints):
        """Initialise a new instance

        :arg npoints: number of 1d quadrature points
        """
        super().__init__()
        assert npoints >= 1
        self._npoints = npoints
        zeta_x, weights_x = np.polynomial.legendre.leggauss(npoints + 1)
        zeta_y, weights_y = np.polynomial.legendre.leggauss(npoints)
        zeta = []
        weights = []
        for x, w_x in zip(zeta_x, weights_x):
            for y, w_y in zip(zeta_y, weights_y):
                x_sq = 1 / 2 * (1 + x)
                y_sq = 1 / 2 * (1 + y)
                zeta.append([x_sq, (1 - x_sq) * y_sq])
                weights.append(1 / 4 * w_x * w_y * (1 - x_sq))
        self._nodes = np.asarray(zeta)
        self._weights = np.asarray(weights)

    @property
    def nodes(self):
        """Return quadrature nodes"""
        return self._nodes

    @property
    def weights(self):
        """Return quadrature weights"""
        return self._weights

    @property
    def degree_of_precision(self):
        """Degree of precision, i.e. highest polynomial degree that
        can be integrated exactly"""
        return 2 * self._npoints - 1
