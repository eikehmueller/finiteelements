import pytest

__all__ = ["element", "polynomial_element"]


def polynomial_element(degree):
    """Return polynomial element of given degree

    Skip the test if the element is not available

    :arg degree: polynomial degree
    """
    try:
        from fem.polynomialelement import PolynomialElement

        return PolynomialElement(degree)
    except:
        pytest.skip(reason="Polynomial element not available")


@pytest.fixture
def element(degree):
    """Return element of given degree

    Skip the test if the element is not available

    :arg degree: polynomial degree
    """
    if degree == 1:
        from fem.linearelement import LinearElement

        return LinearElement()
    else:
        return polynomial_element(degree)
