"""Function space"""

from collections.abc import Iterable


class FunctionSpace:
    """Finite element function space

    Defines association of finite element unknowns with a computational mesh.
    Each unknown has a unique global index i_g = i_g(cell,i) which is related to one (or more) cell
    and a local index. Here we have that:

        0 <= cell < #cells
        0 <= i < 3 * (ndof_per_vertex + ndof_per_facet) + ndof_per_interior
        0 <= i_g < ndof_per_vertex * #vertices + ndof_per_facet * #facets + ndof_per_interior * #cells


    Write n_V = ndof_per_vertex, n_F = ndof_per_facet, n_C = ndof_per_cell and n_total = n_V+n_F+n_C.
    It is assumed that the unknowns are stored in the following order

        * vertex unknowns:
            These dofs have global indices 0 <= i_g < n_V * #vertices and are stored on
            vertices 0,1,2,... in this order, with the unknowns with n_V * V <= i_g < n_V * (V+1)
            all stored on vertex V.
            The local indices are 0 <= i < 3*nV.
        * facet unknowns:
            These dofs have global indices n_V * #vertices <= i_g < n_V * #vertices + n_F * #facet
            and are stored on facets 0,1,2,... in this order, with the unknowns with
            n_V * #vertices + n_F * F <= i_g < n_V * #vertices + n_F * (F+1) all stored on facet F.
            The local indices are 3*n_V <= i < 3*(nV + n_F).
        * cell interior unknowns
            These dofs have global indices
            n_V * #vertices + n_F * #facet <= i_g < n_total
            and are stored in the interiors of cells 0,1,2,... in this order, with the unknowns with
            n_V * #vertices + n_F * #facets + n_C*C <= i_g < n_V * #vertices + n_F * #facets + n_C*(C+1)
            all stored in the interior of cell C.
            The local indices are 3*(n_V+n_F) <= i < 3*(nV + n_F) + n_C.

    """

    def __init__(self, mesh, finiteelement):
        """Initialise a new instance

        :arg mesh: underlying mesh
        :arg finiteelement: underlying finite element
        """
        self.mesh = mesh
        self.finiteelement = finiteelement

    @property
    def ndof(self):
        """Total number of unknowns"""
        return (
            self.mesh.nvertices * self.finiteelement.ndof_per_vertex
            + self.mesh.nfacets * self.finiteelement.ndof_per_facet
            + self.mesh.ncells * self.finiteelement.ndof_per_interior
        )

    def local2global(self, cell, idx):
        """Map local dof-index to global dof-index in a given cell

        The method returns either a single global dof-index if idx is an integer or a list of global
        dof-indices if idx is iterable.

        :arg cell: index of cell
        :arg idx: local dof-index or iterable of local dof-indices
        """
        if isinstance(idx, Iterable):
            return [self._local2global(cell, j) for j in idx]
        else:
            self._local2global(cell, idx)

    def _local2global(self, cell, j):
        """Map a single local dof-index to global dof-index in a given cell

        :arg cell: index of cell
        :arg j: local dof-index
        """
        if j < 3 * self.finiteelement.ndof_per_vertex:
            # dof is associated with vertex v
            v = self.mesh.facet2vertex[
                self.mesh.cell2facet[cell][
                    (j // self.finiteelement.ndof_per_vertex - 1) % 3
                ]
            ][0]
            return v * self.finiteelement.ndof_per_vertex + (
                j % self.finiteelement.ndof_per_vertex
            )

        elif j < 3 * (
            self.finiteelement.ndof_per_vertex + self.finiteelement.ndof_per_facet
        ):
            # dof is associated with facet
            f = self.mesh.cell2facet[cell][
                (j - 3 * self.finiteelement.ndof_per_vertex)
                // self.finiteelement.ndof_per_facet
            ]
            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + f * self.finiteelement.ndof_per_facet
                + (
                    (j - 3 * self.finiteelement.ndof_per_vertex)
                    % self.finiteelement.ndof_per_facet
                )
            )
        else:
            # dof is associated with cell
            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + self.mesh.nfacets * self.finiteelement.ndof_per_facet
                + cell * self.finiteelement.ndof_per_interior
                + j
                - 3
                * (
                    self.finiteelement.ndof_per_vertex
                    + self.finiteelement.ndof_per_facet
                )
            )
