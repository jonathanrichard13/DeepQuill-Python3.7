from typing import Sequence

from ...core import Tensor
from ...internals import type_check

def reshape(x: Tensor, newshape: Sequence[int]) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    # newshape must be a sequence of ints
    type_check(x, "x", Tensor)
    type_check(newshape, "newshape", Sequence, int)

    def grad_fn(child: Tensor) -> None:
        x.grad += child.grad.reshape(x.grad.shape)

    return Tensor(x.nd.reshape(newshape), [x], grad_fn=grad_fn)