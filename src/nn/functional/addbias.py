from cupy import sum, add as _add

from ...classes import Tensor

def addbias(x: Tensor, b: Tensor, axis: int = -1) -> Tensor:
    
    # TYPE CHECKS
    # both x and b must be Tensors
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")
    if not isinstance(b, Tensor):
        raise TypeError(f"{b} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        axes: list[int] = list(range(x.grad.ndim))
        axes.pop(axis)
        x.grad += child.grad
        if len(axes) == 0:
            b.grad += child.grad
        else:
            b.grad += sum(child.grad, axis=axes)
    
    return Tensor((_add(x.nd, b.nd[(..., *[None for _ in range(~axis)]) if axis < 0 else (*[None for _ in range(axis)], ...)])), [x, b], is_leaf=False, grad_fn=grad_fn)