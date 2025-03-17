import numpy as np

__all__ = ["Function", "DualFunction"]


class Function:
    def __init__(self, fs, label="unknown"):
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        return self.functionspace.ndof


class DualFunction:
    def __init__(self, fs, label="unknown"):
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        return self.functionspace.ndof
