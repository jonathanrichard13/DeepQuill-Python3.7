from collections.abc import Sequence
from cupy import amax, argmax, unravel_index, zeros
from numpy import ndarray

from ...classes import Tensor
from ...functions import expr_check, len_check, type_check

def maxpool3d(input_tensor: Tensor, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None) -> Tensor:

    # TYPE CHECKS
    type_check(input_tensor, "input_tensor", Tensor)
    type_check(kernel_size, "kernel_size", (int | Sequence), int)
    type_check(stride, "stride", (int | Sequence | None), int)
    
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
    
    def _maxpool3d(input_nd: ndarray, kernel_size: Sequence[int], stride: Sequence[int]) -> ndarray:

        # get input dimensions
        d_input_nd: tuple[int, int, int] | tuple[int, int, int, int] = input_nd.shape

        # create output image
        output_nd_height: int = ((d_input_nd[-2] - kernel_size[-2]) // stride[-2]) + 1
        output_nd_width: int = ((d_input_nd[-1] - kernel_size[-1]) // stride[-1]) + 1
        output_nd: ndarray = zeros((*d_input_nd[:-2], output_nd_height, output_nd_width))

        # 3D max pooling loop
        for i_output_nd, i_input_nd in zip(range(output_nd_height), range(0, (d_input_nd[-2] - kernel_size[-2] + 1), stride[-2])):
            for j_output_nd, j_input_nd in zip(range(output_nd_width), range(0, (d_input_nd[-1] - kernel_size[-1] + 1), stride[-1])):
                _input_nd: ndarray = input_tensor.nd[..., i_input_nd:(i_input_nd + kernel_size[-2]), j_input_nd:(j_input_nd + kernel_size[-1])]
                output_nd[..., i_output_nd, j_output_nd] = amax(_input_nd, axis=(-1, -2))
        
        return output_nd

    def grad_fn(child: Tensor) -> None:
        d_input: tuple[int, int, int] | tuple[int, int, int, int] = input_tensor.nd.shape
        d_child_grad: tuple[int, int, int] | tuple[int, int, int, int] = child.grad.shape
        if input_tensor.grad.ndim == 3:
            for input_nd_matrix, input_grad_matrix, child_grad_matrix in zip(input_tensor.nd, input_tensor.grad, child.grad):
                for i_child_grad, i_input in zip(range(d_child_grad[-2]), range(0, (d_input[-2] - kernel_size[-2] + 1), stride[-2])):
                    for j_child_grad, j_input in zip(range(d_child_grad[-1]), range(0, (d_input[-1] - kernel_size[-1] + 1), stride[-1])):
                        _input_nd_matrix: ndarray = input_nd_matrix[i_input:(i_input + kernel_size[-2]), j_input:(j_input + kernel_size[-1])]
                        _input_grad_matrix: ndarray = input_grad_matrix[i_input:(i_input + kernel_size[-2]), j_input:(j_input + kernel_size[-1])]
                        _input_grad_matrix[unravel_index(argmax(_input_nd_matrix), _input_nd_matrix.shape)] += child_grad_matrix[i_child_grad, j_child_grad]
        else:
            for input_nd_matrices, input_grad_matrices, child_grad_matrices in zip(input_tensor.nd, input_tensor.grad, child.grad):
                for input_nd_matrix, input_grad_matrix, child_grad_matrix in zip(input_nd_matrices, input_grad_matrices, child_grad_matrices):
                    for i_child_grad, i_input in zip(range(d_child_grad[-2]), range(0, (d_input[-2] - kernel_size[-2] + 1), stride[-2])):
                        for j_child_grad, j_input in zip(range(d_child_grad[-1]), range(0, (d_input[-1] - kernel_size[-1] + 1), stride[-1])):
                            _input_nd_matrix: ndarray = input_nd_matrix[i_input:(i_input + kernel_size[-2]), j_input:(j_input + kernel_size[-1])]
                            _input_grad_matrix: ndarray = input_grad_matrix[i_input:(i_input + kernel_size[-2]), j_input:(j_input + kernel_size[-1])]
                            _input_grad_matrix[unravel_index(argmax(_input_nd_matrix), _input_nd_matrix.shape)] += child_grad_matrix[i_child_grad, j_child_grad]

    return Tensor(_maxpool3d(input_tensor.nd, kernel_size, stride), [input_tensor], grad_fn=grad_fn)