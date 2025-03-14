from collections.abc import Iterable


class FunctionSpace:

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
        """map local dof-index to global dof-index

        :arg cell: index of cell
        :arg idx: local dof-index or iterable of local dof-indices
        """
        if isinstance(idx, Iterable):
            return [self._local2global(cell, j) for j in idx]
        else:
            self._local2global(cell, idx)

    def _local2global(self, cell, j):
        """map local dof-index to global dof-index

        :arg cell: index of cell
        :arg j: local dof-index
        """
        if j < 3 * self.finiteelement.ndof_per_vertex:
            # dof is associated with vertex
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
            f = self.mesh.cell2facet[cell][j // self.finiteelement.ndof_per_facet]
            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + f * self.finiteelement.ndof_per_facet
                + (j % self.finiteelement.ndof_per_facet)
            )
        else:
            # dof is associated with cell
            return (
                self.mesh.nvertices * self.finiteelement.ndof_per_vertex
                + self.mesh.nfacets * self.finiteelement.ndof_per_facet
                + j
            )
