"""Lagrange polynomial finite element"""

from fem.finiteelement import FiniteElement
import numpy as np

__all__ = ["PolynomialElement"]


class PolynomialElement(FiniteElement):
    """Nodal polynomial finite element basis functions of arbitrary degree p in
    two dimensions

    There are (p+1)*(p+2)/2 basis functions, which represent bi-variate
    polynomials P_p(x,y) of degree p on the reference triangle. Each basis function
    is a linear combination of monomials x^a*y^b with a+b <= p. The dofs are
    function evaluations are the nodal points (j*h,k*h) where h=1/p, as shown in the
    following figure.

    Arrangement of the 10 unknowns on reference triangle for p=3:

    V 2
        2
        ! .
        !  .
        5   4
        !     . F 0
    F 1 !      .
        6   9   3
        !         .
        !          .
        0---7---8---1
    V 0       F 2      V 1


    """

    def __init__(self, degree):
        """Initialise new instance

        :arg p: polynomial degree p
        """
        assert degree > 0, "polynomial degree must be >= 1"
        super().__init__()
        self.degree = degree
        # List with powers
        self._powers = [
            [s0, s1 - s0] for s1 in range(self.degree + 1) for s0 in range(s1 + 1)
        ]
        # List with nodal points
        nodal_points = []
        # Spacing of nodal points
        h = 1 / self.degree
        # nodes associated with vertices
        # vertex 0
        nodal_points.append([0, 0])
        # vertex 1
        nodal_points.append([1, 0])
        # vertex 2
        nodal_points.append([0, 1])
        # nodes associated with facets
        # facet 0
        for s in range(1, self.degree):
            nodal_points.append([(self.degree - s) * h, s * h])
        # facet 1
        for s in range(1, self.degree):
            nodal_points.append([0, (self.degree - s) * h])
        # facet 2
        for s in range(1, self.degree):
            nodal_points.append([s * h, 0])
        # nodes associated with interior
        for b in range(1, self.degree - 1):
            for a in range(1, self.degree - b):
                nodal_points.append([a * h, b * h])
        self._nodal_points = np.asarray(nodal_points)
        vandermonde_matrix = self._vandermonde_matrix(self._nodal_points)
        # Solve A.C = Id for the coefficient matrix C. The k-th column of C contains
        # the polynomial coefficients for the k-th basis function
        self._coefficients = np.linalg.inv(vandermonde_matrix)

    def _vandermonde_matrix(self, zeta, grad=False):
        """Construct the Vandermonde matrix or its gradient

        If grad=False, compute the Vandermonde matrix V(zeta)

            V_{r,m}(zeta) = x_r^a(m)*y_r^b(m)

        where the row r corresponds to the index of the point zeta_r = (x_r,y_r) and
        the column m to the power (a(m),b(m)) that the point zeta_r is raised to.
        The resulting matrix has the shape (n, ndof) where npoints is the
        number of points in zeta.

        If grad=True, compute the gradient grad V(zeta) of the Vandermonde matrix
        with

            grad V_{r,m,a} = d V_{r,m}(zeta) / dx_a

        The resulting tensor has the shape (n, ndof,2).

        :arg zeta: array of shape (n, 2) containing the points for which the
                   Vandermonde matrix is to be calculated
        :arg grad: compute gradient?
        """

        npoints = zeta.shape[0]
        if grad:
            mat = np.empty([npoints, len(self._powers), 2])
            for col, s in enumerate(self._powers):
                mat[:, col, 0] = (
                    s[0] * zeta[..., 0] ** max(0, (s[0] - 1)) * zeta[..., 1] ** s[1]
                )
                mat[:, col, 1] = (
                    s[1] * zeta[..., 0] ** s[0] * zeta[..., 1] ** max(0, (s[1] - 1))
                )
        else:
            mat = np.empty([npoints, len(self._powers)])
            for col, s in enumerate(self._powers):
                mat[:, col] = zeta[..., 0] ** s[0] * zeta[..., 1] ** s[1]
        return mat

    @property
    def ndof_per_interior(self):
        """Return number of unknowns associated with the interior of the cell"""
        return ((self.degree - 1) * (self.degree - 2)) // 2

    @property
    def ndof_per_facet(self):
        """Return number of unknowns associated with each facet"""
        return self.degree - 1

    @property
    def ndof_per_vertex(self):
        """Return number of unknowns associated with each vertex"""
        return 1

    def tabulate_dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        Returns vector F with F_ell = lambda_ell(fhat) = fhat(xi_ell) where
        lambda_ell is the ell-th degree of freedom and xi_ell is the corresponding
        nodal point.

        :arg fhat: function fhat to be tabulated
        """
        return fhat(self._nodal_points.T)

    def tabulate(self, zeta):
        """Evaluate all basis functions at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions or a matrix
        of shape (n,ndof) if zeta contains several points.

        :arg zeta: two-dimensional point zeta at which the basis functions are to
                   be evaluated; can also be a matrix of shape (n,2).
        """
        _zeta = np.asarray(zeta)
        mat = np.squeeze(
            self._vandermonde_matrix(
                np.expand_dims(_zeta, axis=list(range(2 - _zeta.ndim))), grad=False
            )
            @ self._coefficients
        )
        return mat

    def tabulate_gradient(self, zeta):
        """Evaluate the gradients of all basis functions at a point inside the
        reference cell

        Returns an matrix of shape (ndof,2) with the evaluation of the gradients of
        all basis functions. If zeta is a matrix containing several points then the
        matrix that is returned is of shape (n,ndof,2)

        :arg zeta: two-dimensional point zeta at which the gradients of the  basis
                functions are to be evaluated; can also be a matrix of shape (n,2).
        """
        _zeta = np.asarray(zeta)
        mat = np.squeeze(
            np.einsum(
                "rma,mj->rja",
                self._vandermonde_matrix(
                    np.expand_dims(_zeta, axis=list(range(2 - _zeta.ndim))), grad=True
                ),
                self._coefficients,
            )
        )
        return mat
