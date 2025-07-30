"""Finite element functions and dual functions (co-functions)"""

import numpy as np

__all__ = ["Function", "CoFunction"]


class Function:
    """Finite element function"""

    def __init__(self, fs, label="unknown"):
        """Initialise new instance

        :arg fs: underlying function space
        :arg label: label for this function
        """
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        """Return total number of unknowns or degrees of freedom"""
        return self.functionspace.ndof


class CoFunction:
    """Finite element co-function, element of dual space"""

    def __init__(self, fs, label="unknown"):
        """Initialise new instance

        :arg fs: underlying function space
        :arg label: label for this function
        """
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        """Return total number of unknowns or degrees of freedom"""
        return self.functionspace.ndof
