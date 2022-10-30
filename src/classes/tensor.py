from typing import Callable
from cupy import ndarray as cdarray, zeros
from numpy import ndarray

from ..functions import type_check

class Tensor:

    def __init__(self, nd: ndarray, parents: list = [], grad_fn: Callable = None, split_idx: int = -1) -> None:

        # TYPE CHECKS
        type_check(nd, "nd", cdarray)
        type_check(parents, "parents", list, Tensor)
        type_check(split_idx, "split_idx", int)
        
        self.nd: ndarray = nd
        self.parents: list[Tensor] = parents
        self.grad_fn: Callable = grad_fn
        self.grad: ndarray = zeros(nd.shape)
        self.split_idx: int = split_idx
        self.backward_countdown: int = 0
        for parent in parents:
            parent.backward_countdown += 1

    def _backward(self):
        self.backward_countdown -= 1
        if self.backward_countdown < 0:
            raise AssertionError(f"Backpropagation error found: too many backward calls to a tensor (backward_countdown={self.backward_countdown}).")
        elif (self.backward_countdown == 0) and (self.grad_fn is not None):
            self.grad_fn(self)
            for parent in self.parents:
                parent._backward()
    
    def _backward_check(self):
        if self.backward_countdown != 0:
            raise AssertionError(f"Backpropagation error found: unexpected number of backward calls to a tensor (backward_countdown={self.backward_countdown})")

    def backward(self):
        self.grad += 1
        self._backward()
        self._backward_check()

    def zero_grad(self):
        self.grad = zeros(self.nd.shape)
        if self.grad_fn is not None:
            for parent in self.parents:
                parent.zero_grad()

    def step(self,lr: float):
        self.nd = self.nd - (self.grad * lr)
        if self.grad_fn is not None:
            for parent in self.parents:
                parent.step(lr)