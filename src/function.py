from functionspace import FunctionSpace


import numpy as np


class Function:
    def __init__(self, fs):
        self.functionspace = fs
        self.data = np.zeros(self.functionspace.ndof)
