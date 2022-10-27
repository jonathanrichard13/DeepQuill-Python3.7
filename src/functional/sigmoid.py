from cupy import apply_over_axes, exp

from ..classes.tensor import Tensor

def sigmoid(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")

    def _sigmoid(x: float) -> float:
        if x >= 0:
            y: float = exp(-x)
            y = 1 / (1 + y)
        else:
            y: float = exp(x)
            y = y / (1 + y)
        return y

    def grad_fn(child: Tensor) -> None:
        # TODO: Check if this is correct
        x.grad += child.grad * child.nd * (1 - child.nd)

    return Tensor(apply_over_axes(_sigmoid, x.nd, list(range(x.nd.ndim))), [x], is_leaf=False, grad_fn=grad_fn)