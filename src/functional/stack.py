from cupy import stack as _stack

from ..classes.tensor import Tensor

def stack(tensors: list[Tensor]) -> Tensor:

    # TYPE CHECKS
    # tensors must be a list of Tensors
    if not isinstance(tensors, list):
        raise TypeError(f"{tensors} is not a list of Tensors.")
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"list element {tensor} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        for i_tensor in range(len(tensors)):
            tensors[i_tensor].grad += child.grad[i_tensor]
    
    tensor_nds: list[Tensor] = []
    for tensor in tensors:
        tensor_nds.append(tensor.nd)

    return Tensor(_stack(tensor_nds), [tensor for tensor in tensors], is_leaf=False, grad_fn=grad_fn)