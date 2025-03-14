from collections.abc import Iterable


class FunctionSpace:

    def __init__(self, mesh, finiteelement):
        """Initialise a new instance

        :arg mesh: underlying mesh
        :arg finiteelement: underlying finite element
        """
        self.mesh = mesh
        self.finiteelement = finiteelement
        self._ndof_per_interior = finiteelement.ndof_per_interior
        self._ndof_per_facet = finiteelement.ndof_per_facet
        self._ndof_per_vertex = finiteelement.ndof_per_vertex
        self._ncells = self.mesh.ncells
        self._nfacets = self.mesh.nfacets
        self._nvertices = self.mesh.nvertices
        self._cell2facet = self.mesh.cell2facet
        self._facet2vertex = self.mesh.facet2vertex

    @property
    def ndof(self):
        """Total number of unknowns"""
        return (
            self._nvertices * self._ndof_per_vertex
            + self._nfacets * self._ndof_per_facet
            + self._ncells * self._ndof_per_interior
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
        if j < 3 * self._ndof_per_vertex:
            # dof is associated with vertex
            v = self._facet2vertex[
                self._cell2facet[cell][(j // self._ndof_per_vertex - 1) % 3]
            ][0]
            return v * self._ndof_per_vertex + (j % self._ndof_per_vertex)

        elif j < 3 * (self._ndof_per_vertex + self._ndof_per_facet):
            # dof is associated with facet
            f = self._cell2facet[cell][j // self._ndof_per_facet]
            return (
                self._nvertices * self._ndof_per_vertex
                + f * self._ndof_per_facet
                + (j % self._ndof_per_facet)
            )
        else:
            # dof is associated with cell
            return (
                self._nvertices * self._ndof_per_vertex
                + self._nfacets * self._ndof_per_facet
                + j
            )
