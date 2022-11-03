from cupy import expand_dims, squeeze as _squeeze

from ...core import Tensor
from ...internals import type_check

def squeeze(x: Tensor, axis: int) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    # axis must be an int
    type_check(x, "x", Tensor)
    type_check(axis, "axis", int)

    def grad_fn(child: Tensor) -> None:
        x.grad += expand_dims(child.grad, axis)

    return Tensor(_squeeze(x.nd, axis=axis), [x], grad_fn=grad_fn)