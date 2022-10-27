from typing import Callable
from numpy import ndarray
from cupy import zeros

class Tensor:

    def __init__(self, nd: ndarray, parents: list = [], is_leaf: bool = True, grad_fn: Callable = None, split_idx: int = -1) -> None:

        # TYPE CHECKS
        if not isinstance(nd, ndarray):
            raise ValueError(f"nd {nd} is not an ndarray.")
        if not isinstance(parents, list):
            raise ValueError(f"Parents {parents} is not a list of Tensors.")
        for parent in parents:
            if not isinstance(parent, Tensor):
                raise ValueError(f"Parent {parent} is not a Tensor.")
        if not isinstance(is_leaf, bool):
            raise ValueError(f"is_leaf {is_leaf} is not a boolean.")
        if not isinstance(split_idx, int):
            raise ValueError(f"split_idx {split_idx} is not an integer.")

        # is_leaf and grad_function are either-or
        if (not is_leaf) and (grad_fn is None):
            raise ValueError("Tensor must be created either as a leaf node or with a gradient function.")
        
        self.nd: ndarray = nd
        self.parents: list[Tensor] = parents
        self.is_leaf: bool = is_leaf
        self.grad_fn: Callable = grad_fn
        self.grad: ndarray = zeros(nd.shape)
        self.split_idx: int = split_idx

    def _backward(self):
        if not self.is_leaf:
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