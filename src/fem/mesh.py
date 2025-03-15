"""Base class for two-dimensional triangular meshes"""

import numpy as np
from matplotlib import pyplot as plt

from fem.linearelement import LinearElement
from fem.vectorelement import VectorElement
from fem.functionspace import FunctionSpace
from fem.function import Function

__all__ = ["Mesh"]


class Mesh:
    """Base class of a two-dimensional mesh consisting of triangular cells

    Stores the mesh topology in the form of adjacency maps and the mesh geometry in the form
    of piecewise line functions.

    Each mesh cell is of the following form:

        V_2
         *
         ! .
         !  .
         !    .                         F
         !     . F_0            V_a ----->----- V_b
     F_1 v      ^
         !   C    .
         !         .
         !          .
         *---->------*
     V_0       F_2      V_1

    The facets are oriented in a counterclockwise orientation. The topology is represented by
    storing the indices [F_0(C),F_1(C),F_2(C)] (in this order) of the facets of a given cell C
    in a list cell2facet

        cell2facet = [[F_0(0),F_1(0),F_2(0)], [F_0(1),F_1(1),F_2(1)],...]

    and the indices [V_a(F), V_b(F)] (in this order) of the vertices that define a given facet F
    in a list facet2vertex

        facet2vertex = [[V_a(0),V_b(0)],[V_a(1),V_b(1)],...]

    Note that the three vertices of a given cell C can be obtained as

        V_0(C) = facet2vertex[cell2facet[C][2]]
        V_1(C) = facet2vertex[cell2facet[C][0]]
        V_2(C) = facet2vertex[cell2facet[C][1]]

    This class should not be instantiated since it does not contain any cells, use one of the meshes
    in utilitymesh.py instead.
    """

    def __init__(self):
        """Initialise a new instance"""
        self.vertices = None
        self.cell2facet = None
        self.facet2vertex = None
        self.vertices = None
        self.coordinates = None

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

    def _initialise_coordinates(self):
        """Initialise the piecewise linear coordinate field

        The coordinate field is constructed from the coordinates of the mesh vertices,
        this method needs to be called after refinement.
        """
        coord_fs = FunctionSpace(self, VectorElement(LinearElement()))
        self.coordinates = Function(coord_fs, "coordinates")
        for dim in (0, 1):
            self.coordinates.data[dim::2] = self.vertices[:, dim]

    def refine(self, nref=1):
        """Refine the mesh multiple times by subdividing each triangle

        Repeatedly sub-divides each mesh cell into four smaller similar smaller triangles

        :arg nref: number of refinement steps
        """
        for _ in range(nref):
            self._refine()
        self._initialise_coordinates()

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
            fine_facet2vertex.append([new_vertex, v2])
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
            for j in range(2, -1, -1):
                fine_facet2vertex.append(
                    [facet_centre_vertex[(j + 1) % 3], facet_centre_vertex[j]]
                )
            # add interior cells
            # three cells that touch coarse facets
            for j in range(3):
                fine_cell2facet.append(
                    [
                        coarse2finefacet[coarse_facets[(2 + j) % 3]][1],
                        coarse2finefacet[coarse_facets[j]][0],
                        facet_idx + (3 - j) % 3,
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

    def visualise(self, filename):
        """Plot the mesh and save to disk

        :arg filename: name of file to plot to
        """
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        # points
        axs[0, 0].plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            linewidth=0,
            marker="o",
            markersize=4,
            color="blue",
        )
        for vertex in range(self.nvertices):
            axs[0, 0].annotate(
                f"{vertex:3d}",
                (self.vertices[vertex, :]),
                fontsize=6,
            )
        # facets
        for facet in range(self.nfacets):
            p = np.asarray(
                [self.vertices[vertex] for vertex in self.facet2vertex[facet]]
            )
            m = np.mean(p, axis=0)
            rho = 0.8
            p = rho * p + (1 - rho) * np.expand_dims(m, axis=0)
            axs[0, 1].plot(
                p[:, 0], p[:, 1], linewidth=2, marker="o", markersize=4, color="blue"
            )
            axs[0, 1].annotate(
                f"{facet:3d}",
                (m[0], m[1]),
                fontsize=6,
            )
            axs[0, 1].arrow(
                p[0, 0],
                p[0, 1],
                m[0] - p[0, 0],
                m[1] - p[0, 1],
                width=0,
                linewidth=0,
                head_width=0.05,
                color="blue",
            )
        # cells
        for cell in range(self.ncells):
            p = np.zeros((4, 3))
            for j, facet in enumerate(self.cell2facet[cell]):
                p[j, :2] = np.asarray(self.vertices[self.facet2vertex[facet][0]])
            p[-1, :] = p[0, :]
            orientation = np.cross(p[1, :] - p[0, :], p[2, :] - p[1, :])[2]
            orientation /= np.abs(orientation)
            m = np.mean(p[:-1, :], axis=0)
            rho = 0.8
            p = rho * p + (1 - rho) * np.expand_dims(m, axis=0)
            axs[1, 0].plot(
                p[:, 0],
                p[:, 1],
                linewidth=2,
                color="blue" if orientation > 0 else "red",
            )
            axs[1, 0].annotate(
                f"{cell:3d}",
                (m[0], m[1]),
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=6,
            )
        plt.savefig(filename, bbox_inches="tight")
