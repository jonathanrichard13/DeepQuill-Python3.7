from typing import Sequence

from ...classes import Tensor
from ...functions import type_check

def reshape(x: Tensor, newshape: tuple[int]) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    # newshape must be a tuple of ints
    type_check(x, "x", Tensor)
    type_check(newshape, "newshape", Sequence, int)

    def grad_fn(child: Tensor) -> None:
        x.grad += child.grad.reshape(x.grad.shape)

    return Tensor(x.nd.reshape(newshape), [x], grad_fn=grad_fn)