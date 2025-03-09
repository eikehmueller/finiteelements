import numpy as np
from matplotlib import pyplot as plt

from abc import ABC, abstractmethod


class Mesh2d(ABC):
    def __init__(self):
        self._vertices = None
        self._cell2facet = None
        self._facet2vertex = None

    @property
    def vertices(self):
        return np.array(self._vertices)

    @property
    def cell2facet(self):
        return list(self._cell2facet)

    @property
    def facet2vertex(self):
        return list(self._facet2vertex)

    @property
    def ncells(self):
        return len(self._cell2facet)

    @property
    def nfacets(self):
        return len(self._facet2vertex)

    @property
    def nvertices(self):
        return self._vertices.shape[0]

    def refine(self, nref=1):
        for _ in range(nref):
            self._refine()

    def _refine(self):
        # Pointer to current vertex
        vertex_idx = self.nvertices
        # Pointer to current fine facet
        facet_idx = 0
        # make space for new vertices
        self._vertices = np.pad(self._vertices, ((0, self.nfacets), (0, 0)))
        # STEP 1: refine all edges and add new vertices
        fine_facet2vertex = []
        coarse2finefacet = []
        for coarse_facet in range(self.nfacets):
            new_vertex = vertex_idx
            v1, v2 = self.facet2vertex[coarse_facet]
            self._vertices[new_vertex, :] = 0.5 * (
                self._vertices[v1] + self._vertices[v2]
            )
            fine_facet2vertex.append([v1, new_vertex])
            fine_facet2vertex.append([new_vertex, v2])
            coarse2finefacet.append([facet_idx, facet_idx + 1])
            vertex_idx += 1
            facet_idx += 2
        fine_cell2facet = []
        # STEP 2: refine all cells
        for coarse_cell in range(self.ncells):
            coarse_facets = self.cell2facet[coarse_cell]
            print(coarse_cell, coarse_facets)
            for coarse_facet in coarse_facets:
                print("  ", coarse_facet, coarse2finefacet[coarse_facet])
            # vertices on coarse facet centres
            facet_centre_vertex = [
                fine_facet2vertex[coarse2finefacet[coarse_facet][0]][1]
                for coarse_facet in coarse_facets
            ]
            print("facet_centre_vertex", facet_centre_vertex)
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
        self._cell2facet = fine_cell2facet
        self._facet2vertex = fine_facet2vertex

    def visualise(self, filename):
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
            p = np.empty((4, 2))
            for j, facet in enumerate(self.cell2facet[cell]):
                p[j, :] = np.asarray(self.vertices[self.facet2vertex[facet][0]])
            p[-1, :] = p[0, :]
            orientation = np.cross(p[1, :] - p[0, :], p[2, :] - p[1, :])
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


class RectangleMesh(Mesh2d):
    def __init__(self, Lx=1.0, Ly=1.0, nref=0):
        super().__init__()
        self._Lx = Lx
        self._Ly = Ly
        self._vertices = np.asarray(
            [[0, 0], [self._Lx, 0], [0, self._Ly], [self._Lx, self._Ly]], dtype=float
        )
        self._cell2facet = [[0, 1, 2], [3, 1, 4]]
        self._facet2vertex = [[0, 1], [1, 2], [2, 0], [3, 1], [2, 3]]
        self.refine(nref)

    @property
    def Lx(self):
        return self._Lx

    @property
    def Ly(self):
        return self._Ly


mesh = RectangleMesh()
mesh.refine(2)
print(f"#cells = {mesh.ncells}, #facets = {mesh.nfacets}, #vertices = {mesh.nvertices}")
print(mesh.cell2facet)
print(mesh.facet2vertex)
mesh.visualise("rectangle_mesh.pdf")
