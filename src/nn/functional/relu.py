from cupy import maximum

from ...classes import Tensor

def relu(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x.grad[x.nd > 0] += child.grad[x.nd > 0]

    return Tensor(maximum(x.nd, 0), [x], is_leaf=False, grad_fn=grad_fn)