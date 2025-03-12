import numpy as np


class Quadrature2d:
    """Base class for quadrature in 2d"""

    def __init__(self):
        """Initialise a new instance"""
        self._xi = []
        self._weights = []

    @property
    def xi(self):
        """return quadrature points"""
        return self._xi

    @property
    def weights(self):
        """Return quadrature weights"""
        return self._weights


class GaussLegendreQuadrature2d(Quadrature2d):
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
        self._xi = np.asarray(xi)
        self._weights = np.asarray(weights)


quad = GaussLegendreQuadrature2d(4)
print(quad.xi)
print(quad.weights)
