from typing import List
from cupy import split as _split

from ...classes import Tensor
from ...functions import type_check

def split(tensor: Tensor, indices_or_sections: int, axis: int = 0) -> List[Tensor]:

    # TYPE CHECKS
    # tensor must be a Tensor
    # axis must be an int
    type_check(tensor, "tensor", Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        stride: int = tensor.nd.shape[axis] // indices_or_sections
        if axis < 0:
            _axis: int = ~axis
            tensor.grad[(*(slice(None) for _ in range(child.grad.ndim - _axis - 1)), slice(child.split_idx, child.split_idx + stride), *(slice(None) for _ in range(_axis)))] += child.grad
        else:
            tensor.grad[(*(slice(None) for _ in range(axis)), slice(child.split_idx, child.split_idx + stride), *(slice(None) for _ in range(child.grad.ndim - axis - 1)))] += child.grad

    return [Tensor(nd, [tensor], grad_fn=grad_fn, split_idx=i_nd) for i_nd, nd in enumerate(_split(tensor.nd, indices_or_sections, axis))]