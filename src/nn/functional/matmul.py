from numpy import ndarray
from cupy import matmul as _matmul, sum, swapaxes

from ...classes import Tensor

def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # both x1 and x2 must be Tensors
    if not isinstance(x1, Tensor):
        raise TypeError(f"{x1} is not a Tensor.")
    if not isinstance(x2, Tensor):
        raise TypeError(f"{x2} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        _x1_grad: ndarray = _matmul(child.grad, swapaxes(x2.nd, -1, -2))
        _x2_grad: ndarray = _matmul(swapaxes(x1.nd, -1, -2), child.grad)
        x1_ndim: int = x1.nd.ndim
        x2_ndim: int = x2.nd.ndim
        if x1_ndim == x2_ndim:
            x1.grad += _x1_grad
            x2.grad += _x2_grad
        elif x1_ndim < x2_ndim:
            x1.grad += sum(_x1_grad, axis=list(range(x2_ndim - x1_ndim)))
            x2.grad += _x2_grad
        else:
            x1.grad += _x1_grad
            x2.grad += sum(_x2_grad, axis=list(range(x1_ndim - x2_ndim)))

    return Tensor((_matmul(x1.nd, x2.nd)), [x1, x2], is_leaf=False, grad_fn=grad_fn)