"""Linear finite element"""

import numpy as np
from fem.finiteelement import FiniteElement

__all__ = ["LinearElement"]


class LinearElement(FiniteElement):
    """Linear finite element basis functions in 2d

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
        self._nodal_points = np.asarray([[0, 0], [1, 0], [0, 1]])

    @property
    def ndof_per_interior(self):
        """Return number of unknowns associated with the interior of the cell"""
        return 0

    @property
    def ndof_per_facet(self):
        """Return number of unknowns associated with each facet"""
        return 0

    @property
    def ndof_per_vertex(self):
        """Return number of unknowns associated with each vertex"""
        return 1

    def tabulate_dofs(self, fhat):
        """Evaluate the dofs on a given function on the reference element

        :arg fhat: function fhat defined for 2d vectors
        """
        return fhat(self._nodal_points)

    def tabulate(self, zeta):
        """Evaluate all basis functions at points inside the reference cell

        Returns a vector of length 3 with the evaluation of all three basis functions.
        If zeta is a matrix of shape (npoints,2) whose columns represent points, then
        an array of shape (npoints,3) will be returned

        :arg zeta: two-dimensional point zeta at which the basis functions are to be
                evaluated, can also be an array of shape (n,2) whose columns are the points.
        """
        return np.asarray(
            [1 - zeta[..., 0] - zeta[..., 1], zeta[..., 0], zeta[..., 1]]
        ).T

    def tabulate_gradient(self, zeta):
        """Evaluate the gradients of all basis function at point inside the reference cell

        Returns an matrux of shape (3,2) with the evaluation of the gradients of all three
        basis functions. If zeta is a matrix of shape (npoints,2) whose columns represent points, then
        an array of shape (npoints,3,2)

        :arg zeta: two-dimensional point zeta at which the gradients of the basis
                functions are to be evaluated, can also be an array of shape (n,2)
                whose columns are the points.
        """
        if zeta.ndim == 1:
            return np.asarray([[-1, -1], [1, 0], [0, 1]])
        else:
            return np.repeat(
                np.expand_dims(
                    np.asarray([[-1, -1], [1, 0], [0, 1]]),
                    axis=list(range(zeta.ndim - 1)),
                ),
                [zeta.shape[0]],
                axis=0,
            )
