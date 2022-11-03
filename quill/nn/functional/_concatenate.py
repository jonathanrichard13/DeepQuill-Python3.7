from collections.abc import Sequence
from cupy import concatenate as _concatenate, split
from numpy import ndarray

from ...classes import Tensor
from ...functions import type_check

def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:

    # TYPE CHECKS
    # tensors must be a sequence of Tensors
    # axis must be an int
    type_check(tensors, "tensors", Sequence, Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        _child_grad: list[ndarray] = [_grad for _grad in split(child.grad, len(tensors), axis)]
        for i_tensor in range(len(tensors)):
            tensors[i_tensor].grad += _child_grad[i_tensor]

    return Tensor(_concatenate([tensor.nd for tensor in tensors], axis), [tensor for tensor in tensors], grad_fn=grad_fn)