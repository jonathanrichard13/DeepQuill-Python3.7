from cupy import add as _add

from ..classes.tensor import Tensor

def add(x1: Tensor, x2: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # both x1 and x2 must be Tensors
    if not isinstance(x1, Tensor):
        raise TypeError(f"{x1} is not a Tensor.")
    if not isinstance(x2, Tensor):
        raise TypeError(f"{x2} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x1.grad += child.grad
        x2.grad += child.grad

    return Tensor((_add(x1.nd, x2.nd)), [x1, x2], is_leaf=False, grad_fn=grad_fn)