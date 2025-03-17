import numpy as np

__all__ = ["Function", "CoFunction"]


class Function:
    def __init__(self, fs, label="unknown"):
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        return self.functionspace.ndof


class CoFunction:
    def __init__(self, fs, label="unknown"):
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
        self.label = label

    @property
    def ndof(self):
        return self.functionspace.ndof
