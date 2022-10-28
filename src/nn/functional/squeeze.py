from cupy import expand_dims, squeeze as _squeeze

from ...classes.tensor import Tensor

def squeeze(x: Tensor, axis: int) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")
    # axis must be an int
    if not isinstance(axis, int):
        raise TypeError(f"{axis} is not an int.")

    def grad_fn(child: Tensor) -> None:
        x.grad += expand_dims(child.grad, axis)

    return Tensor(_squeeze(x.nd, axis=axis), [x], is_leaf=False, grad_fn=grad_fn)