from fem.quadrature import GaussLegendreQuadrature
import pytest
import numpy as np


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_quadrature2d_weight_sum(degree):
    """Check that weights sum up to 1/2"""
    quadrature = GaussLegendreQuadrature(degree)
    tolerance = 1.0e-9
    assert abs(np.sum(quadrature.weights) - 1 / 2) < tolerance


def test_quadrature2d_quartic_integration():
    """Check that the function x^2*(1-x)^2*y^2*(1-y)^2 is integrated exactly"""
    poly = lambda xi: xi[0] ** 2 * (1 - xi[0]) ** 2 * xi[1] ** 2 * (1 - xi[1]) ** 2
    quadrature = GaussLegendreQuadrature(3)
    integral = 0
    for xi, weight in zip(quadrature.nodes, quadrature.weights):
        integral += poly(xi) * weight
    tolerance = 1.0e-9
    assert abs(integral - 1 / 2 * 1 / 30**2) < tolerance
