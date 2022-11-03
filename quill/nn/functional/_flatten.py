from ...classes import Tensor
from ...internals import type_check

def flatten(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    type_check(x, "x", Tensor)

    def grad_fn(child: Tensor) -> None:
        x.grad += child.grad.reshape(x.grad.shape)

    return Tensor(x.nd.flatten(), [x], grad_fn=grad_fn)