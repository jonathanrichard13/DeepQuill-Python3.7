from typing import List
from cupy import split, squeeze

from ...classes import Tensor
from ...functions import type_check

def unstack(tensor: Tensor, axis: int = 0) -> List[Tensor]:

    # TYPE CHECKS
    # tensor must be a Tensor
    type_check(tensor, "tensor", Tensor)

    def grad_fn(child: Tensor) -> None:
        if axis < 0:
            _axis: int = ~axis
            tensor.grad[(*(slice(None) for _ in range(child.grad.ndim - _axis - 1)), child.split_idx, *(slice(None) for _ in range(_axis)))] += child.grad
        else:
            tensor.grad[(*(slice(None) for _ in range(axis)), child.split_idx, *(slice(None) for _ in range(child.grad.ndim - axis - 1)))] += child.grad

    return [Tensor(squeeze(nd, axis=axis), [tensor], grad_fn=grad_fn, split_idx=i_nd) for i_nd, nd in enumerate(split(tensor.nd, tensor.nd.shape[axis], axis))]