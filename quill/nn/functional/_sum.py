from cupy import expand_dims, sum as _sum, repeat

from ...classes import Tensor
from ...internals import type_check

def sum(x: Tensor, axis: int = 0) -> Tensor:
    
    # TYPE CHECKS
    # x must be a Tensor
    # axis must be an int
    type_check(x, "x", Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        x.grad += repeat(expand_dims(child.grad, axis), x.grad.shape[axis], axis)

    return Tensor((_sum(x.nd, axis=axis)), [x], grad_fn=grad_fn)