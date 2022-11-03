from ...core import Tensor
from ...internals import type_check

def add(x1: Tensor, x2: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # both x1 and x2 must be Tensors
    type_check(x1, "x1", Tensor)
    type_check(x2, "x2", Tensor)

    def grad_fn(child: Tensor) -> None:
        x1.grad += child.grad
        x2.grad += child.grad

    return Tensor(x1.nd + x2.nd, [x1, x2], grad_fn=grad_fn)