from fem.quadrature import (
    GaussLegendreQuadratureLineSegment,
    GaussLegendreQuadratureReferenceTriangle,
)
import pytest
import numpy as np
import math


@pytest.mark.parametrize("npoints", [1, 2, 3, 4])
def test_weight_sum(npoints):
    """Check that weights sum up to 1/2"""
    quadrature = GaussLegendreQuadratureReferenceTriangle(npoints)
    tolerance = 1.0e-9
    assert abs(np.sum(quadrature.weights) - 1 / 2) < tolerance


def test_quartic_integration():
    """Check that the function x^2*(1-x)^2*y^2*(1-y)^2 is integrated exactly"""
    poly = lambda xi: xi[0] ** 2 * (1 - xi[0]) ** 2 * xi[1] ** 2 * (1 - xi[1]) ** 2
    quadrature = GaussLegendreQuadratureReferenceTriangle(3)
    integral = 0
    for xi, weight in zip(quadrature.nodes, quadrature.weights):
        integral += poly(xi) * weight
    tolerance = 1.0e-9
    assert abs(integral - 1 / 2 * 1 / 30**2) < tolerance


@pytest.mark.parametrize("npoints", [1, 2, 3, 4, 5])
def test_exact_integration(npoints):
    """Check that all monomials up to 2*npoints+1 are integrated exactly"""
    quadrature = GaussLegendreQuadratureReferenceTriangle(npoints)
    dop = quadrature.degree_of_precision
    error = []
    for z0 in range(dop + 1):
        for z1 in range(dop + 1 - z0):
            s_numerical = 0
            for w, xi in zip(quadrature.weights, quadrature.nodes):
                s_numerical += w * xi[0] ** z0 * xi[1] ** z1
            s_exact = (
                math.factorial(z0) * math.factorial(z1) / math.factorial(z0 + z1 + 2)
            )
            error.append(s_numerical - s_exact)
    tolerance = 1.0e-9
    assert np.allclose(error, 0, rtol=tolerance)


@pytest.mark.parametrize("npoints", [1, 2, 3, 4])
def test_weight_sum_line(npoints):
    """Check that weights sum up to ||b-a||"""
    v_a = np.asarray([0.2, 0.7])
    v_b = np.asarray([0.8, 1.4])
    quadrature = GaussLegendreQuadratureLineSegment(v_a, v_b, npoints)
    tolerance = 1.0e-9
    assert abs(np.sum(quadrature.weights) - np.linalg.norm(v_b - v_a)) < tolerance


def test_quadratic_integration_line():
    """Check that the function x^2+y^2 is integrated exactly along a line segment"""
    poly = lambda xi: xi[0] ** 2 + xi[1] ** 2
    v_a = np.asarray([0.2, 0.7])
    v_b = np.asarray([0.8, 1.4])
    quadrature = GaussLegendreQuadratureLineSegment(v_a, v_b, 2)
    integral = 0
    for xi, weight in zip(quadrature.nodes, quadrature.weights):
        integral += poly(xi) * weight
    tolerance = 1.0e-9
    assert (
        abs(
            integral
            - np.linalg.norm(v_b - v_a)
            * (np.dot(v_a, v_b) + 1 / 3 * np.dot(v_b - v_a, v_b - v_a))
        )
        < tolerance
    )
