"""Base class for two-dimensional triangular meshes"""

from functools import cached_property
import numpy as np

from fem.linearelement import LinearElement
from fem.vectorelement import VectorElement
from fem.functionspace import FunctionSpace
from fem.function import Function

__all__ = ["Mesh"]


class Mesh:
    """Class representing a two-dimensional mesh consisting of triangular cells

    Stores the mesh topology in the form of adjacency maps and the mesh geometry
    in the form piecewise linear functions.

    Each mesh cell is of the following form:

        v 2
         *
         ! .
         !  .
         !    .                         F
         !     . F 0            V a ----->----- V b
     F 1 v      ^
         !   C    .
         !         .
         !          .
         *---->------*
     v 0       F 2      v 1

    The facets are oriented in a counterclockwise orientation. The topology is
    represented by storing the indices [F_0(C),F_1(C),F_2(C)] (in this order) of the
    facets of a given cell C in a list cell2facet

        cell2facet = [[F_0(0),F_1(0),F_2(0)], [F_0(1),F_1(1),F_2(1)],...]

    and the indices [V_a(F), V_b(F)] (in this order) of the vertices that define a given facet F
    in a list facet2vertex

        facet2vertex = [[V_a(0),V_b(0)],[V_a(1),V_b(1)],...]

    The three vertices of a given cell C can be obtained as

        V_0(C) = facet2vertex[cell2facet[C][2]]
        V_1(C) = facet2vertex[cell2facet[C][0]]
        V_2(C) = facet2vertex[cell2facet[C][1]]

    and they are stored in another list given by

        cell2vertex = [[V_0(0),V_1(0),V_2(0)],[V_0(1),V_1(1),V_2(1)],...]

    which is derived from cell2facet and facet2cell.

    This class should not be instantiated since it does not contain any cells, use one of the meshes
    in utilitymesh.py instead.
    """

    def __init__(self, vertices, cell2facet, facet2vertex):
        """Initialise a new instance"""
        self.vertices = vertices
        self.cell2facet = cell2facet
        self.facet2vertex = facet2vertex
        self.coordinates = self._initialise_coordinates()

    @property
    def ncells(self):
        """Number of cells of the mesh"""
        return len(self.cell2facet)

    @property
    def nfacets(self):
        """Number of facets of the mesh"""
        return len(self.facet2vertex)

    @property
    def nvertices(self):
        """Number of vertices of the mesh"""
        return self.vertices.shape[0]

    @cached_property
    def cell2vertex(self):
        """Return mapping from cells to associated vertices"""
        return [
            [
                (
                    set(self.facet2vertex[self.cell2facet[cell][(rho + 1) % 3]])
                    & set(self.facet2vertex[self.cell2facet[cell][(rho + 2) % 3]])
                ).pop()
                for rho in range(3)
            ]
            for cell in range(self.ncells)
        ]

    def _initialise_coordinates(self):
        """Initialise the piecewise linear coordinate field

        The coordinate field is constructed from the coordinates of the mesh vertices,
        this method needs to be called after refinement.
        """
        coord_fs = FunctionSpace(self, VectorElement(LinearElement()))
        coordinates = Function(coord_fs, "coordinates")
        for dim in (0, 1):
            coordinates.data[dim::2] = self.vertices[:, dim]
        return coordinates

    def refine(self, nref=1):
        """Refine the mesh multiple times by subdividing each triangle

        Repeatedly sub-divides each mesh cell into four smaller similar smaller triangles

        :arg nref: number of refinement steps
        """
        for _ in range(nref):
            self._refine()
        self.coordinates = self._initialise_coordinates()

    def _refine(self):
        """Refine the mesh by once subdividing each triangle

        Sub-divides each mesh cell into four smaller similar smaller triangles."""

        # Pointer to current vertex
        vertex_idx = self.nvertices
        # Pointer to current fine facet
        facet_idx = 0
        # make space for new vertices
        self.vertices = np.pad(self.vertices, ((0, self.nfacets), (0, 0)))
        # STEP 1: refine all edges and add new vertices
        fine_facet2vertex = []
        coarse2finefacet = []
        for coarse_facet in range(self.nfacets):
            new_vertex = vertex_idx
            v1, v2 = self.facet2vertex[coarse_facet]
            self.vertices[new_vertex, :] = 0.5 * (self.vertices[v1] + self.vertices[v2])
            fine_facet2vertex.append([v1, new_vertex])
            fine_facet2vertex.append([v2, new_vertex])
            coarse2finefacet.append([facet_idx, facet_idx + 1])
            vertex_idx += 1
            facet_idx += 2
        fine_cell2facet = []
        # STEP 2: refine all cells
        for coarse_cell in range(self.ncells):
            coarse_facets = self.cell2facet[coarse_cell]
            # vertices on coarse facet centres
            facet_centre_vertex = [
                fine_facet2vertex[coarse2finefacet[coarse_facet][0]][1]
                for coarse_facet in coarse_facets
            ]
            # add interior facets
            for rho in range(3):
                fine_facet2vertex.append(
                    sorted(
                        [
                            facet_centre_vertex[(rho + 1) % 3],
                            facet_centre_vertex[(rho + 2) % 3],
                        ]
                    )
                )
            # add interior cells

            coarse2finefacet[coarse_facets[rho]][0]

            # fine facets on boundary of coarse cell
            boundary_fine_facets = []
            for rho in range(3):
                fine_facets = coarse2finefacet[coarse_facets[rho]]
                if (
                    fine_facet2vertex[fine_facets[0]][0]
                    == self.cell2vertex[coarse_cell][(rho + 1) % 3]
                ):
                    boundary_fine_facets.append(fine_facets)
                else:
                    boundary_fine_facets.append(fine_facets[::-1])
            # three cells that touch coarse facets
            for rho in range(3):
                fine_cell2facet.append(
                    [
                        boundary_fine_facets[(rho + 2) % 3][1],
                        boundary_fine_facets[rho][0],
                        facet_idx + (rho + 1) % 3,
                    ]
                )
            # cell that does not touch any coarse facet
            fine_cell2facet.append(
                [
                    facet_idx,
                    facet_idx + 1,
                    facet_idx + 2,
                ]
            )
            facet_idx += 3
        self.cell2facet = fine_cell2facet
        self.facet2vertex = fine_facet2vertex
        # delete property to reset cache
        del self.cell2vertex
