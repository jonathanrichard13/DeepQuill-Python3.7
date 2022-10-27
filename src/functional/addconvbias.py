from cupy import sum, add as _add

from ..classes.tensor import Tensor

def addconvbias(x: Tensor, bias: Tensor) -> Tensor:
    
    # TYPE CHECKS
    # both x and bias must be Tensors
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")
    if not isinstance(bias, Tensor):
        raise TypeError(f"{bias} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x.grad += child.grad
        bias.grad += sum(child.grad, axis=(-1, -2))

    return Tensor((_add(x.nd, bias.nd[..., None, None])), [x, bias], is_leaf=False, grad_fn=grad_fn)