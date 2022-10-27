from cupy import matmul as _matmul

from ..classes.tensor import Tensor

def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # both x1 and x2 must be Tensors
    if not isinstance(x1, Tensor):
        raise TypeError(f"{x1} is not a Tensor.")
    if not isinstance(x2, Tensor):
        raise TypeError(f"{x2} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x1.grad += _matmul(child.grad, x2.nd.T)
        x2.grad += _matmul(x1.nd.T, child.grad)

    return Tensor((_matmul(x1.nd, x2.nd)), [x1, x2], is_leaf=False, grad_fn=grad_fn)