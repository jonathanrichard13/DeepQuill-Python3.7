from cupy import expand_dims as _expand_dims, squeeze

from ...classes.tensor import Tensor

def expand_dims(x: Tensor, axis: int) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")
    # axis must be an int
    if not isinstance(axis, int):
        raise TypeError(f"{axis} is not an int.")

    def grad_fn(child: Tensor) -> None:
        x.grad += squeeze(child.grad, axis=axis)

    return Tensor(_expand_dims(x.nd, axis), [x], is_leaf=False, grad_fn=grad_fn)