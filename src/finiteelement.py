"""Basis functions"""

from abc import ABC, abstractmethod
import numpy as np


class FiniteElement2d(ABC):
    """Base class for two-dimensional finite element basis functions on the reference triangle

    The vertices and facets and of the reference triangle are arranged as in the following
    diagram:

        V 2
         *
         ! .
         !  .
         !    .
         !     . F 0
     F 1 !      .
         !        .
         !         .
         !          .
         *-----------*
     V 0       F 2      V 1

     Facet unknowns are arranged in a counter-clockwise order.
    """

    def __init__(self):
        """Initialise new instance"""
        self._ndof_per_vertex = 0
        self._ndof_per_facet = 0
        self._ndof_per_cell = 0

    @abstractmethod
    def evaluate(self, xi):
        """Evaluate all basis function at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """

    @abstractmethod
    def evaluate_gradient(self, xi):
        """Evaluate the gradient of all basis functions at a point inside the reference cell

        Returns an vector of shape (ndof,2) with the evaluation of the gradients of all
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be evaluated.
        """

    @abstractmethod
    def dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        :arg fhat: function fhat(xhat) where xhat is a two-dimensional vector
        """

    @property
    def ndof(self):
        """Return number of unknowns

        Need to set the variable _ndof in derived classes
        """
        return 3 * (self._ndof_per_vertex + self._ndof_per_facet) + self._ndof_per_cell

    @property
    def ndof_per_cell(self):
        """Return number of unknowns associated with the interior of the cell"""
        return self._ndof_per_cell

    @property
    def ndof_per_facet(self):
        """Return number of unknowns associated with each facet"""
        return self._ndof_per_facet

    @property
    def ndof_per_vertex(self):
        """Return number of unknowns associated with each vertex"""
        return self._ndof_per_vertex


class LinearFiniteElement2d(FiniteElement2d):
    """Linear finite element basis functions in two dimensions

    There are 3 basis functions, which represent bi-variant linear functions on the reference
    triangle:

        phi_0(x,y) = 1 - x - y
        phi_1(x,y) = x
        phi_2(x,y) = y

    The dofs are function evaluations are the nodal points which coincide with the
    vertices, as shown in the following figure.

    Arrangement of the 3 unknowns on reference triangle:

    V 2
        2
        ! .
        !  .
        !   .
        !     . F 0
    F 1 !      .
        !       .
        !         .
        !          .
        0-----------1
    V 0       F 2      V 1


    """

    def __init__(self):
        """Initialise new instance"""
        super().__init__()
        self._ndof_per_vertex = 1
        self._nodal_points = [[0, 0], [1, 0], [0, 1]]

    def dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        :arg fhat: function fhat(xhat) where xhat is a two-dimensional vector
        """
        dof_vector = np.empty(3)
        for j in range(3):
            dof_vector[j] = fhat(np.asarray(self._nodal_points[j]))
        return dof_vector

    def evaluate(self, xi):
        """Evaluate all basis functions at a point inside the reference cell

        Returns a vector of length 3 with the evaluation of all three basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """
        x, y = xi
        return np.asarray([1 - x - y, x, y])

    def evaluate_gradient(self, xi):
        """Evaluate the gradients of all basis function at a point inside the reference cell

        Returns an vector of shape (3,2) with the evaluation of the gradients of all three
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be evaluated.
        """
        return np.asarray([[-1, -1], [1, 0], [0, 1]])


class PolynomialFiniteElement2d(FiniteElement2d):
    """Nodal polynomial finite element basis functions of arbitrary degree p in two dimensions

    There are (p+1)*(p+2)/2 basis functions, which represent bi-variate polynomials P_p(x,y)
    of degree p on the reference triangle. Each basis function is a linear combination of
    monomials x^a*y^b with a+b <= p. The dofs are function evaluations are the nodal points
    (j*h,k*h) where h=1/p, as shown in the following figure.

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
        self._degree = degree
        # List with powers
        self._powers = []
        # List with nodal points
        self._nodal_points = []
        # Spacing of nodal points
        h = 1 / self._degree
        self._ndof_per_vertex = 1
        self._ndof_per_facet = self._degree - 1
        self._ndof_per_cell = ((self._degree - 1) * (self._degree - 2)) // 2

        # nodes associated with vertices
        # vertex 0
        self._powers.append([0, 0])
        self._nodal_points.append([0, 0])
        # vertex 1
        self._powers.append([self._degree, 0])
        self._nodal_points.append([1, 0])
        # vertex 2
        self._powers.append([0, self._degree])
        self._nodal_points.append([0, 1])
        # nodes associated with facets
        # facet 0
        for j in range(1, self._degree):
            self._powers.append([self.degree - j, j])
            self._nodal_points.append([(self.degree - j) * h, j * h])
        # facet 1
        for j in range(1, self._degree):
            self._powers.append([0, self.degree - j])
            self._nodal_points.append([0, (self.degree - j) * h])
        # facet 2
        for j in range(1, self._degree):
            self._powers.append([j, 0])
            self._nodal_points.append([j * h, 0])
        # nodes associated with interior
        for b in range(1, self._degree - 1):
            for a in range(1, self._degree - b):
                self._powers.append([a, b])
                self._nodal_points.append([a * h, b * h])
        # Construct the matrix A such that
        #    A_{row,col} = x_j^a*y_k^b
        # where the row corresponds to the index of the nodal point (x_j,x_k) and
        # the column to the power (a,b) that the nodal point is raised to
        A = np.empty([len(self._nodal_points), len(self._powers)])
        for row, (x, y) in enumerate(self._nodal_points):
            for col, (a, b) in enumerate(self._powers):
                A[row, col] = x**a * y**b
        # Solve A.C = Id for the coefficient matrix C. The k-th column of C contains
        # the polynomial coefficients for the k-th basis function
        self._coefficients = np.linalg.inv(A)
        self._ndof = ((self._degree + 1) * (self._degree + 2)) // 2

    @property
    def degree(self):
        return self._degree

    def dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        :arg fhat: function fhat(xhat) where xhat is a two-dimensional vector
        """
        dof_vector = np.empty(self.ndof)
        for j in range(self.ndof):
            dof_vector[j] = fhat(np.asarray(self._nodal_points[j]))
        return dof_vector

    def evaluate(self, xi):
        """Evaluate all basis functions at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """

        x, y = xi
        value = np.zeros(self.ndof)
        for k in range(self.ndof):
            for coefficient, (a, b) in zip(self._coefficients[:, k], self._powers):
                value[k] += coefficient * x**a * y**b
        return value

    def evaluate_gradient(self, xi):
        """Evaluate the gradients of all basis functions at a point inside the reference cell

        Returns an vector of shape (ndof,2) with the evaluation of the gradients of all
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be evaluated.
        """

        x, y = xi
        grad = np.zeros((self.ndof, 2))
        for k in range(self.ndof):
            for coefficient, (a, b) in zip(self._coefficients[:, k], self._powers):
                if a > 0:
                    grad[k, 0] += coefficient * a * x ** (a - 1) * y**b
                if b > 0:
                    grad[k, 1] += coefficient * b * x**a * y ** (b - 1)
        return grad


class VectorElement(FiniteElement2d):
    def __init__(self, finiteelement):
        super().__init__()
        self._finiteelement = finiteelement
        self._ndof_per_vertex = 2 * finiteelement._ndof_per_vertex
        self._ndof_per_facet = 2 * finiteelement._ndof_per_facet
        self._ndof_per_cell = 2 * finiteelement._ndof_per_cell

    def evaluate(self, xi):
        """Evaluate all basis functions at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """
        scalar_evaluate = self._finiteelement.evaluate(xi)
        value = np.zeros((self.ndof, 2))

        offset = 0
        for ndof_entity in (
            self._ndof_per_vertex,
            self._ndof_per_facet,
            self._ndof_per_cell,
        ):
            for dim in (0, 1):
                value[offset + dim : offset + dim + ndof_entity : 2, dim] = (
                    scalar_evaluate[offset // 2 : (offset + ndof_entity) // 2]
                )
            offset += ndof_entity
        return value

    def evaluate_gradient(self, xi):
        """Evaluate the gradients of all basis functions at a point inside the reference cell

        Returns an vector of shape (ndof,2,2) with the evaluation of the gradients of all
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be evaluated.
        """
        scalar_grad = self._finiteelement.evaluate_grad(xi)
        grad = np.zeros((self.ndof, 2, 2))

        offset = 0
        for ndof_entity in (
            self._ndof_per_vertex,
            self._ndof_per_facet,
            self._ndof_per_cell,
        ):
            for dim in (0, 1):
                grad[offset + dim : offset + dim + ndof_entity : 2, dim, :] = (
                    scalar_grad[offset // 2 : (offset + ndof_entity) // 2, :]
                )
            offset += ndof_entity
        return grad

    def dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        :arg fhat: vector-valued function fhat(xhat) where xhat is a two-dimensional vector
        """
        dof_vector = np.empty(self.ndof)
        for dim in range(0, 1):
            dof_vector[dim::2] = self._finiteelement.dofs(lambda xhat: fhat(xhat)[dim])
