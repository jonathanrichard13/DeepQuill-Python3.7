from cupy import expand_dims as _expand_dims, squeeze

from ...classes import Tensor
from ...functions import type_check

def expand_dims(x: Tensor, axis: int) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    # axis must be an int
    type_check(x, "x", Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        x.grad += squeeze(child.grad, axis=axis)

    return Tensor(_expand_dims(x.nd, axis), [x], is_leaf=False, grad_fn=grad_fn)