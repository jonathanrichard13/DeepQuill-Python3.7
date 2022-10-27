from cupy import expand_dims, sum as _sum, repeat

from ..classes.tensor import Tensor

def sum(x: Tensor, axis: int = 0) -> Tensor:
    
    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")
    # axis must be an int
    if not isinstance(axis, int):
        raise TypeError(f"{axis} is not an int.")

    def grad_fn(child: Tensor) -> None:
        x.grad += repeat(expand_dims(child.grad, axis), x.grad.shape[axis], axis)

    return Tensor((_sum(x.nd, axis=axis)), [x], is_leaf=False, grad_fn=grad_fn)