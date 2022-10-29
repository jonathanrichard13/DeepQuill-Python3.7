from cupy import split, squeeze

from ...classes import Tensor
from ...functions import type_check

def unstack(tensor: Tensor) -> list[Tensor]:

    # TYPE CHECKS
    # tensor must be a Tensor
    type_check(tensor, "tensor", Tensor)

    def grad_fn(child: Tensor) -> None:
        tensor.grad[child.split_idx] += child.grad
        tensor.backward_countdown -= 1
    
    n_tensors = tensor.nd.shape[0]
    tensor_nds: list[Tensor] = split(tensor.nd, n_tensors, 0)

    tensor.backward_countdown += n_tensors
    return [Tensor(squeeze(tensor_nds[i_tensor], axis=0), [tensor], is_leaf=False, grad_fn=grad_fn, split_idx=i_tensor) for i_tensor in range(n_tensors)]