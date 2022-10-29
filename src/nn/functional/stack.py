from collections.abc import Collection
from cupy import split, squeeze, stack as _stack
from numpy import ndarray

from ...classes import Tensor
from ...functions import type_check

def stack(tensors: Collection[Tensor], axis: int = 0) -> Tensor:

    # TYPE CHECKS
    # tensors must be a list of Tensors
    # axis must be an int
    type_check(tensors, "tensors", Collection, Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        _child_grad: list[ndarray] = [squeeze(_grad, axis) for _grad in split(child.grad, child.grad.shape[axis], axis)]
        for i_tensor in range(len(tensors)):
            tensors[i_tensor].grad += _child_grad[i_tensor]

    return Tensor(_stack([tensor.nd for tensor in tensors], axis), [tensor for tensor in tensors], grad_fn=grad_fn)