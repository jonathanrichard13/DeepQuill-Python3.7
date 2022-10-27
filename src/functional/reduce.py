from cupy import sum, stack

from ..classes.tensor import Tensor

def reduce(x: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x.grad += stack([child.grad for _ in range(x.grad.shape[0])])

    return Tensor((sum(x.nd, axis=0)), [x], is_leaf=False, grad_fn=grad_fn)