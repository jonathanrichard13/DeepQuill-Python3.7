from numpy import ndarray
from cupy import mean, reciprocal, zeros

from ...classes.tensor import Tensor

def avgpool3d(input_tensor: Tensor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> Tensor:

    # TYPE CHECKS
    if not isinstance(input_tensor, Tensor):
        raise TypeError(f"Input tensor {input_tensor} is not a Tensor.")
    if (not isinstance(kernel_size, int)) and (not isinstance(kernel_size, tuple)):
        raise TypeError(f"Parameter kernel_size ({kernel_size}) must be either an integer or a tuple.")
    if (stride is not None) and (not isinstance(stride, int)) and (not isinstance(stride, tuple)):
        raise TypeError(f"Parameter stride ({stride}) must be one of the following: None, integer, or a tuple.")
    
    # cast kernel_size and stride into tuple
    if isinstance(kernel_size, int):
        kernel_size: tuple[int, int] = (kernel_size, kernel_size)
    if stride is None:
        stride: tuple[int, int] = kernel_size
    elif isinstance(stride, int):
        stride: tuple[int, int] = (stride, stride)
    
    # MISMATCHED DIMENSION CHECKS
    if (input_tensor.nd.ndim != 3) and (input_tensor.nd.ndim != 4):
        raise IndexError(f"Input tensor is {input_tensor.nd.ndim}-dimensional instead of 3-or-4-dimensional.")
    if len(kernel_size) != 2:
        raise IndexError(f"If kernel_size ({kernel_size}) is a tuple, it must be of length = 2.")
    if len(stride) != 2:
        raise IndexError(f"If stride ({stride}) is a tuple, it must be of length = 2.")
    
    def _avgpool3d(input_nd: ndarray, kernel_size: tuple[int, int], stride: tuple[int, int]) -> ndarray:

        # get input dimensions
        d_input_nd: tuple[int, int, int] | tuple[int, int, int, int] = input_nd.shape

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
        d_input_grad: tuple[int, int, int] | tuple[int, int, int, int] = input_tensor.grad.shape
        d_child_grad: tuple[int, int, int] | tuple[int, int, int, int] = child.grad.shape
        for i_child_grad, i_input_grad in zip(range(d_child_grad[-2]), range(0, (d_input_grad[-2] - kernel_size[-2] + 1), stride[-2])):
            for j_child_grad, j_input_grad in zip(range(d_child_grad[-1]), range(0, (d_input_grad[-1] - kernel_size[-1] + 1), stride[-1])):
                input_tensor.grad[..., i_input_grad:(i_input_grad + kernel_size[-2]), j_input_grad:(j_input_grad + kernel_size[-1])] += child.grad[..., None, None, i_child_grad, j_child_grad] * reciprocal(float(kernel_size[-2] * kernel_size[-1]))

    return Tensor(_avgpool3d(input_tensor.nd, kernel_size, stride), [input_tensor], is_leaf=False, grad_fn=grad_fn)