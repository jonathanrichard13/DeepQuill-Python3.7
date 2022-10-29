from typing import Callable
from cupy import ndarray as cdarray, zeros
from numpy import ndarray

from ..functions import type_check

class Tensor:

    def __init__(self, nd: ndarray, parents: list = [], is_leaf: bool = True, grad_fn: Callable = None, split_idx: int = -1) -> None:

        # TYPE CHECKS
        type_check(nd, "nd", cdarray)
        type_check(parents, "parents", list, Tensor)
        type_check(is_leaf, "is_leaf", bool)
        type_check(split_idx, "split_idx", int)

        # is_leaf and grad_function are either-or
        if (not is_leaf) and (grad_fn is None):
            raise ValueError("Tensor must be created either as a leaf node or with a gradient function.")
        
        self.nd: ndarray = nd
        self.parents: list[Tensor] = parents
        self.is_leaf: bool = is_leaf
        self.grad_fn: Callable = grad_fn
        self.grad: ndarray = zeros(nd.shape)
        self.split_idx: int = split_idx
        self.backward_countdown: int = 0

    def _backward(self):
        if not (self.is_leaf or (self.backward_countdown > 0)):
            if self.grad_fn is not None:
                self.grad_fn(self)
            for parent in self.parents:
                parent._backward()
    
    def backward(self):
        self.grad += 1
        self._backward()

    def zero_grad(self):
        self.grad = zeros(self.nd.shape)
        if not self.is_leaf:
            for parent in self.parents:
                parent.zero_grad()

    def step(self,lr: float):
        self.nd = self.nd - (self.grad * lr)
        if not self.is_leaf:
            for parent in self.parents:
                parent.step(lr)