"""Base class for finite element basis functions"""

from abc import ABC, abstractmethod
from functools import cache

__all__ = ["FiniteElement"]


class FiniteElement(ABC):
    """Abstract base class for 2d finite element on reference triangle

    The vertices and facets and of the reference triangle are arranged
    as in the following diagram:

        v 2
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
     v 0       F 2      v 1

     Unknowns associated with vertices and facets are arranged in a
     counter-clockwise order.
    """

    def __init__(self):
        """Initialise new instance"""

    @abstractmethod
    def tabulate(self, zeta):
        """Evaluate all basis functions at one or several points
        inside reference cell

        Returns a vector of length ndof with the evaluation of all basis functions or a matrix
        of shape (n,ndof) if zeta contains several points.

        :arg zeta: two-dimensional point zeta at which the basis functions are to be
                evaluated; can also be a matrix of shape (n,2) in which case the
                evaluations at all n points are returned.
        """

    @abstractmethod
    def tabulate_gradient(self, zeta):
        """Evaluate gradients of all basis functions at one or several points
        inside the reference cell

        Returns an matrix of shape (ndof,2) with the evaluation of the gradients of all
        basis functions. If zeta is a matrix containing several points then the matrix that is
        returned is of shape (n,ndof,2)

        :arg zeta: two dimensional point zeta at which the gradients of the  basis functions
                are to be evaluated; can also be a matrix of shape (n,2) in which case
                the gradient evaluations at all n points are returned
        """

    @abstractmethod
    def tabulate_dofs(self, fhat):
        """Tabulate the dofs on a given function on the reference element

        Returns vector F with F_ell = lambda_ell(fhat) lambda_ell the ell-th
        degree of freedom.

        :arg fhat: function fhat to be tabulated
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
    def dofmap(self, entity_type, rho, j):
        """Compute dof-index of j-th degree of freedom associated with rho-th entity

        Returns index in the range 0, 1, ..., ndof-1

        :arg entity_type: type of topological entity. Must be "vertex", "facet" or "interior"
        :arg rho: index of entity, ignored if entity_type is "interior"
        :arg j: index of dof on entity
        """
        if entity_type == "vertex":
            return self.ndof_per_vertex * rho + j
        elif entity_type == "facet":
            return 3 * self.ndof_per_vertex + self.ndof_per_facet * rho + j
        elif entity_type == "interior":
            return 3 * (self.ndof_per_vertex + self.ndof_per_facet) + j
        else:
            raise RuntimeError(f"Unknown entity type: {entity_type}")

    @cache
    def inverse_dofmap(self, ell):
        """Work out entity and local index on entity for a given degree of freedom

        Returns a tuple (entity_type,rho,j) where entity_type is the type of topological
        entity associated with the dof, rho is the index of this entity and j is
        the index of the dof on that entity.

        :arg ell: index of degree of freedom
        """
        if ell < 3 * self.ndof_per_vertex:
            # dof is associated with vertex v
            return ("vertex", ell // self.ndof_per_vertex, ell % self.ndof_per_vertex)
        elif ell < 3 * (self.ndof_per_vertex + self.ndof_per_facet):
            # dof is associated with facet
            ell = ell - 3 * self.ndof_per_vertex
            return ("facet", ell // self.ndof_per_facet, ell % self.ndof_per_facet)
        elif ell < self.ndof:
            return (
                "interior",
                0,
                ell - 3 * (self.ndof_per_vertex + self.ndof_per_facet),
            )
        else:
            raise RuntimeError(f"dof-index {ell} is larger than ndof-1 = {self.ndof-1}")
