from cupy import apply_along_axis, exp

from ...classes.tensor import Tensor

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

    return Tensor(apply_along_axis(_sigmoid, -1, x.nd.reshape(-1, 1)).reshape(x.nd.shape), [x], is_leaf=False, grad_fn=grad_fn)