from typing import Sequence, Tuple, Union
from cupy import mean, reciprocal, zeros
from numpy import ndarray

from ...core import Tensor
from ...internals import expr_check, len_check, type_check

def avgpool3d(input_tensor: Tensor, kernel_size: Union[int, Sequence[int]], stride: Union[int, Sequence[int], None] = None) -> Tensor:

    # TYPE CHECKS
    type_check(input_tensor, "input_tensor", Tensor)
    type_check(kernel_size, "kernel_size", (int, Sequence), int)
    type_check(stride, "stride", (int, Sequence, type(None)), int)
    
    # cast kernel_size and stride into tuple
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    # MISMATCHED DIMENSION CHECKS
    if (input_tensor.nd.ndim != 3) and (input_tensor.nd.ndim != 4):
        raise IndexError(f"Input tensor is {input_tensor.nd.ndim}-dimensional instead of 3-or-4-dimensional.")
    len_check(kernel_size, "kernel_size", 2)
    len_check(stride, "stride", 2)

    # VALUE OUT OF RANGE CHECKS
    for i_kernel_size in range(len(kernel_size)):
        expr_check(kernel_size[i_kernel_size], f"kernel_size[{i_kernel_size}]", lambda x: x > 0)
    for i_stride in range(len(stride)):
        expr_check(stride[i_stride], f"stride[{i_stride}]", lambda x: x > 0)
    
    def _avgpool3d(input_nd: ndarray, kernel_size: Sequence[int], stride: Sequence[int]) -> ndarray:

        # get input dimensions
        d_input_nd: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = input_nd.shape

        # create output image
        output_nd_height: int = ((d_input_nd[-2] - kernel_size[-2]) // stride[-2]) + 1
        output_nd_width: int = ((d_input_nd[-1] - kernel_size[-1]) // stride[-1]) + 1
        output_nd: ndarray = zeros((*d_input_nd[:-2], output_nd_height, output_nd_width))

        # 3D average pooling loop
        for i_output_nd, i_input_nd in zip(range(output_nd_height), range(0, (d_input_nd[-2] - kernel_size[-2] + 1), stride[-2])):
            for j_output_nd, j_input_nd in zip(range(output_nd_width), range(0, (d_input_nd[-1] - kernel_size[-1] + 1), stride[-1])):
                output_nd[..., i_output_nd, j_output_nd] = mean(input_nd[..., i_input_nd:(i_input_nd + kernel_size[-2]), j_input_nd:(j_input_nd + kernel_size[-1])], axis=(-1, -2))
        
        return output_nd

    def grad_fn(child: Tensor) -> None:
        d_input_grad: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = input_tensor.grad.shape
        d_child_grad: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = child.grad.shape
        for i_child_grad, i_input_grad in zip(range(d_child_grad[-2]), range(0, (d_input_grad[-2] - kernel_size[-2] + 1), stride[-2])):
            for j_child_grad, j_input_grad in zip(range(d_child_grad[-1]), range(0, (d_input_grad[-1] - kernel_size[-1] + 1), stride[-1])):
                input_tensor.grad[..., i_input_grad:(i_input_grad + kernel_size[-2]), j_input_grad:(j_input_grad + kernel_size[-1])] += child.grad[..., None, None, i_child_grad, j_child_grad] * reciprocal(float(kernel_size[-2] * kernel_size[-1]))

    return Tensor(_avgpool3d(input_tensor.nd, kernel_size, stride), [input_tensor], grad_fn=grad_fn)