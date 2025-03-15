"""Vector finite element"""

from fem.finiteelement import FiniteElement
import numpy as np

__all__ = ["VectorElement"]


class VectorElement(FiniteElement):
    """Vector finite element in 2d

    The element is constructed by taking the product of two copies of an underlying
    finite element, arranging the degrees of freedom in contiguous order as shown in
    the following example for which the underlying element is a
    PolynomialFiniteElement2d of degree p=3:

       V 2
       4,5
        ! .
        !  .
        !   .
        !    .
     10,11    8,9
        !      .
        !       . F 0
    F 1 !        .
        !    18   .
     12,13   19    6,7
        !           .
        !            .
        !             .
        !    14    16  .
       0,1---15----17---2,3
    V 0       F 2      V 1

    Dofs with even/odd indices correspond to the vector components in the two
    dimensions respectively.

    """

    def __init__(self, finiteelement):
        """Initialise new instance

        :arg finitelement: underlying finite element
        """
        super().__init__()
        self._finiteelement = finiteelement

    @property
    def ndof_per_interior(self):
        """Return number of unknowns associated with the interior of the cell"""
        return 2 * self._finiteelement.ndof_per_interior

    @property
    def ndof_per_facet(self):
        """Return number of unknowns associated with each facet"""
        return 2 * self._finiteelement.ndof_per_facet

    @property
    def ndof_per_vertex(self):
        """Return number of unknowns associated with each vertex"""
        return 2 * self._finiteelement.ndof_per_vertex

    def tabulate_dofs(self, fhat):
        """Tabulate all dofs on a given function on the reference element

        :arg fhat: vector-valued function fhat(xhat) where xhat is a two-dimensional vector
        """
        dof_vector = np.empty(self.ndof)
        for dim in (0, 1):
            dof_vector[dim::2] = self._finiteelement.tabulate_dofs(
                lambda xhat, d=dim: fhat(xhat)[d]
            )
        return dof_vector

    def tabulate(self, xi):
        """Tabulate all basis functions at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """
        scalar_tabulation = self._finiteelement.tabulate(xi)
        value = np.zeros((self.ndof, 2))
        for dim in (0, 1):
            value[dim::2, dim] = scalar_tabulation[:]
        return value

    def tabulate_gradient(self, xi):
        """Tabulate the gradients of all basis functions at a point inside the reference cell

        Returns an vector of shape (ndof,2,2) with the evaluation of the gradients of all
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be evaluated.
        """
        scalar_grad = self._finiteelement.evaluate_grad(xi)
        grad = np.zeros((self.ndof, 2, 2))

        for dim in (0, 1):
            grad[dim::2, dim, :] = scalar_grad[:, :]
        return grad
