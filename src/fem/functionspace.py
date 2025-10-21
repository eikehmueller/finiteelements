"""Finite element function space

Function spaces are defined by a combination of a mesh and a finite element
"""

from collections.abc import Iterable
from fem.vectorelement import VectorElement


class FunctionSpace:
    """Finite element function space

    Defines association of finite element unknowns with a computational mesh.
    Each unknown has a unique global index ell_g = ell_g(alpha,ell) which is
    related to one (or more) cell with index alpha and a local index ell.

    Here we have that:

        0 <= alpha < #cells
        0 <= ell < 3 * (ndof_per_vertex + ndof_per_facet) + ndof_per_interior
        0 <= ell_g < ndof_per_vertex * #vertices
                     + ndof_per_facet * #facets
                     + ndof_per_interior * #cells


    Let n_V = ndof_per_vertex, n_F = ndof_per_facet, n_C = ndof_per_cell and
    n_total = n_V+n_F+n_C. It is assumed that the unknowns are stored in the following
    order:

        * vertex unknowns:
            These dofs have global indices 0 <= ell_g < n_V * #vertices and are stored on
            vertices 0,1,2,... in this order, with the unknowns with n_V * V <= ell_g < n_V * (V+1)
            all stored on vertex V.
            The local indices are 0 <= ell < 3*nV.
        * facet unknowns:
            These dofs have global indices n_V * #vertices <= ell_g < n_V * #vertices + n_F * #facet
            and are stored on facets 0,1,2,... in this order, with the unknowns with
            n_V * #vertices + n_F * F <= ell_g < n_V * #vertices + n_F * (F+1) all stored on facet F.
            The local indices are 3*n_V <= ell < 3*(nV + n_F).
        * cell interior unknowns
            These dofs have global indices
            n_V * #vertices + n_F * #facet <= ell_g < n_total
            and are stored in the interiors of cells 0,1,2,... in this order, with the unknowns with
            n_V * #vertices + n_F * #facets + n_C*C <= ell_g < n_V * #vertices + n_F * #facets + n_C*(C+1)
            all stored in the interior of cell C.
            The local indices are 3*(n_V+n_F) <= ell < 3*(nV + n_F) + n_C.

    """

    def __init__(self, mesh, finiteelement):
        """Initialise a new instance

        :arg mesh: underlying mesh
        :arg finiteelement: underlying finite element
        """
        self.mesh = mesh
        self.finiteelement = finiteelement
        # For vector-elements, each degree of freedom is associated with more than
        # one scalar unknown. This is important when considering facet orientations.
        self.n_components = 2 if type(self.finiteelement) is VectorElement else 1

    @property
    def ndof(self):
        """Total number of unknowns"""
        return (
            self.mesh.nvertices * self.finiteelement.ndof_per_vertex
            + self.mesh.nfacets * self.finiteelement.ndof_per_facet
            + self.mesh.ncells * self.finiteelement.ndof_per_interior
        )

    def local2global(self, alpha, ell):
        """Map local dof-index to global dof-index in a given cell

        The method returns either a single global dof-index if idx is an integer or a list of global
        dof-indices if idx is iterable.

        :arg alpha: index of cell
        :arg ell: local dof-index or iterable of local dof-indices
        """
        if isinstance(ell, Iterable):
            return [self._local2global(alpha, _ell) for _ell in ell]
        else:
            self._local2global(alpha, ell)

    def _local2global(self, alpha, ell):
        """Map a single local dof-index to global dof-index in a given cell

        :arg alpha: index of cell
        :arg ell: local dof-index
        """
        entity_type, rho, j = self.finiteelement.inverse_dofmap(ell)
        if entity_type == "vertex":
            # dof is associated with vertex v
            gamma = self.mesh.cell2vertex[alpha][rho]
            return gamma * self.finiteelement.ndof_per_vertex + j

        elif entity_type == "facet":
            # dof is associated with facet
            beta = self.mesh.cell2facet[alpha][rho]
            # Check whether facet is oriented in the same direction as the local facet
            # If this is not the case, we need to reverse the order in which the
            # unknowns are accessed
            _j = (
                j
                if (
                    self.mesh.facet2vertex[beta][0]
                    == self.mesh.cell2vertex[alpha][(rho + 1) % 3]
                )
                else (
                    self.n_components
                    * (
                        self.finiteelement.ndof_per_facet // self.n_components
                        - j // self.n_components
                        - 1
                    )
                    + j % self.n_components
                )
            )

            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + beta * self.finiteelement.ndof_per_facet
                + _j
            )
        else:
            # dof is associated with cell
            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + self.mesh.nfacets * self.finiteelement.ndof_per_facet
                + alpha * self.finiteelement.ndof_per_interior
                + j
            )
