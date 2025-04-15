"""Base class for finite element basis functions"""

from abc import ABC, abstractmethod
from functools import cache

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
    def tabulate(self, zeta):
        """Evaluate all basis functions at a point inside the reference cell

        Returns a vector of length ndof with the evaluation of all basis functions or a matrix
        of shape (npoints,ndof) if zeta contains several points.

        :arg zeta: point zeta=(x,y) at which the basis functions are to be evaluated; can also be a
                 matrix of shape (npoints,2).
        """

    @abstractmethod
    def tabulate_gradient(self, zeta):
        """Evaluate the gradients of all basis functions at a point inside the reference cell

        Returns an matrix of shape (ndof,2) with the evaluation of the gradients of all
        basis functions. If zeta is a matrix containing several points then the matrix that is
        returned is of shape (npoints,ndof,2)

        :arg zeta: point zeta=(x,y) at which the gradients of the  basis functions are to be evaluated;
                 can also be a matrix of shape (npoints,2).
        """

    @abstractmethod
    def tabulate_dofs(self, fhat):
        """Tabulate the dofs on a given function on the reference element

        :arg fhat: function fhat defined for 2d vectors
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

    @cache
    def dofmap(self, entity_type, i, k):
        """Compute dof-index of j-th degree of freedom associated with i-th entity

        Returns index in the range 0, 1, ..., ndof-1

        :arg entity_type: type of topological entity. Must be "vertex", "facet" or "interior"
        :arg i: index of entity, ignored if entity_type is "interior"
        :arg k: index of dof on entity
        """
        if entity_type == "vertex":
            return self.ndof_per_vertex * i + k
        elif entity_type == "vertex":
            return 3 * self.ndof_per_vertex + self.ndof_per_facet * i + k
        elif entity_type == "interior":
            return 3 * (self.ndof_per_vertex + self.ndof_per_facet) + k
        else:
            raise RuntimeError(f"Unknown entity type: {entity_type}")

    @cache
    def inverse_dofmap(self, j):
        """Work out entity and local index on entity for a given degree of freedom

        Returns a tuple (entity_type,i,k) where entity_type is the type of topological
        entity associated with the dof, i is the index of this entity and k is the index of
        the dof on that entity.

        :arg j: index of degree of freedom
        """
        if j < 3 * self.ndof_per_vertex:
            # dof is associated with vertex v
            return ("vertex", j // self.ndof_per_vertex, j % self.ndof_per_vertex)
        elif j < 3 * (self.ndof_per_vertex + self.ndof_per_facet):
            # dof is associated with facet
            ell = j - 3 * self.ndof_per_vertex
            return ("facet", ell // self.ndof_per_facet, ell % self.ndof_per_facet)
        elif j < self.ndof:
            return ("interior", 0, j - 3 * (self.ndof_per_vertex + self.ndof_per_facet))
        else:
            raise RuntimeError(f"dof-index {j} is larger than ndof-1 = {self.ndof-1}")
