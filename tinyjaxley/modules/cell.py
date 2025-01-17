from .base import Module

class Cell(Module):
    def __init__(self, branches = None, parents = None):
        super().__init__(branches)
        self.parents = parents