from numpy import ndarray
from cupy import split, squeeze, stack as _stack

from ...classes.tensor import Tensor

def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:

    # TYPE CHECKS
    # tensors must be a list of Tensors
    if not isinstance(tensors, list):
        raise TypeError(f"{tensors} is not a list of Tensors.")
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"list element {tensor} is not a Tensor.")
    # axis must be an int
    if not isinstance(axis, int):
        raise TypeError(f"{axis} is not an int.")

    def grad_fn(child: Tensor) -> None:
        child_grads: list[ndarray] = [squeeze(nd, axis) for nd in split(child.grad, child.grad.shape[axis], axis)]
        for i_tensor in range(len(tensors)):
            tensors[i_tensor].grad += child_grads[i_tensor]

    return Tensor(_stack([tensor.nd for tensor in tensors], axis), [tensor for tensor in tensors], is_leaf=False, grad_fn=grad_fn)