from cupy import maximum

from ...core import Tensor
from ...internals import type_check

def relu(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    type_check(x, "x", Tensor)

    def grad_fn(child: Tensor) -> None:
        x.grad[x.nd > 0] += child.grad[x.nd > 0]

    return Tensor(maximum(x.nd, 0), [x], grad_fn=grad_fn)