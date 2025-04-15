from fem.quadrature import (
    GaussLegendreQuadratureLineSegment,
    GaussLegendreQuadratureReferenceTriangle,
)
import pytest
import numpy as np


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_weight_sum(degree):
    """Check that weights sum up to 1/2"""
    quadrature = GaussLegendreQuadratureReferenceTriangle(degree)
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


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_weight_sum_line(degree):
    """Check that weights sum up to ||b-a||"""
    v_a = np.asarray([0.2, 0.7])
    v_b = np.asarray([0.8, 1.4])
    quadrature = GaussLegendreQuadratureLineSegment(v_a, v_b, degree)
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
