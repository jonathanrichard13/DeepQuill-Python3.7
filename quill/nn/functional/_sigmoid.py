from cupy import apply_along_axis, exp

from ...classes import Tensor
from ...internals import type_check

def sigmoid(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    type_check(x, "x", Tensor)

    def _sigmoid(x: float) -> float:
        if x >= 0:
            y: float = exp(-x)
            y = 1 / (1 + y)
        else:
            y: float = exp(x)
            y = y / (1 + y)
        return y

    def grad_fn(child: Tensor) -> None:
        x.grad += (child.nd * (1 - child.nd)) * child.grad

    return Tensor(apply_along_axis(_sigmoid, -1, x.nd.reshape(-1, 1)).reshape(x.nd.shape), [x], grad_fn=grad_fn)