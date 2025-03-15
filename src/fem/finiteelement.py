"""Base class for finite element basis functions"""

from abc import ABC, abstractmethod

__all__ = ["FiniteElement"]


class FiniteElement(ABC):
    """Abstract base class for 2d finite element basis functions on the reference triangle

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

     Unknowns associated with vertices and facets are assumed to to be continuous and
     arranged in a counter-clockwise order.
    """

    def __init__(self):
        """Initialise new instance"""

    @abstractmethod
    def tabulate(self, xi):
        """Tabulate all basis function at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions.

        :arg xi: point xi=(x,y) at which the basis functions are to be evaluated.
        """

    @abstractmethod
    def tabulate_gradient(self, xi):
        """Tabulate the gradient of all basis functions at a point inside the reference cell

        Returns an vector of shape (ndof,2) with the evaluation of the gradients of all
        basis functions.

        :arg xi: point xi=(x,y) at which the gradients of the basis functions are to be
                 evaluated.
        """

    @abstractmethod
    def tabulate_dofs(self, fhat):
        """Tabulate the dofs on a given function on the reference element

        :arg fhat: function fhat(xhat) where xhat is a two-dimensional vector
        """

    @property
    @abstractmethod
    def ndof_per_interior(self):
        """Return number of unknowns associated with the interior of the cell"""

    @property
    @abstractmethod
    def ndof_per_facet(self):
        """Return number of unknowns associated with each facet"""

    @property
    @abstractmethod
    def ndof_per_vertex(self):
        """Return number of unknowns associated with each vertex"""

    @property
    def ndof(self):
        """Return total number of unknowns"""
        return 3 * (self.ndof_per_vertex + self.ndof_per_facet) + self.ndof_per_interior
