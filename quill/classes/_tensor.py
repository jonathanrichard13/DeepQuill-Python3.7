from typing import Callable, List, Union
from typing_extensions import Self
from cupy import ndarray as cdarray, zeros
from numpy import ndarray

from ..functions import type_check

class Tensor:

    def __init__(self, nd: ndarray, parents: List[Self] = [], grad_fn: Callable[[Self], None] = None, split_idx: int = -1) -> None:

        # TYPE CHECKS
        type_check(nd, "nd", cdarray)
        type_check(parents, "parents", list, Tensor)
        type_check(split_idx, "split_idx", int)
        
        self.nd: ndarray = nd
        self.parents: List[Tensor] = parents
        self.grad_fn: Callable = grad_fn
        self.grad: ndarray = zeros(nd.shape)
        self.split_idx: int = split_idx
        self.i_backward: int = 0
        self.n_backward: int = 0
        for parent in parents:
            parent.n_backward += 1
        self.velocity: Union[ndarray, None] = None
    
    def _zero_i_backward(self) -> None:
        if self.i_backward != 0:
            self.i_backward = 0
            if self.grad_fn is not None:
                for parent in self.parents:
                    parent._zero_i_backward()

    def _backward(self) -> None:
        self.i_backward += 1
        if self.i_backward > self.n_backward:
            raise AssertionError(f"Backpropagation error found: too many backward calls to a tensor (i_backward={self.i_backward}, n_backward={self.n_backward}).")
        elif (self.i_backward == self.n_backward) and (self.grad_fn is not None):
            self.grad_fn(self)
            for parent in self.parents:
                parent._backward()

    def backward(self) -> None:
        self._zero_i_backward()
        _n_backward: int = self.n_backward
        self.n_backward = 1
        self.grad += 1
        self._backward()
        self.n_backward = _n_backward