from numpy import ndarray
from cupy import amax, arange, argmax, int64, repeat, tile, zeros

from ..classes.tensor import Tensor

def maxpool3d(input_tensor: Tensor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> Tensor:

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
    
    def _maxpool3d(input_nd: ndarray, kernel_size: tuple[int, int], stride: tuple[int, int], return_indices: bool = False) -> ndarray | tuple[ndarray, ndarray]:

        # get input dimensions
        d_input_nd: tuple[int, int, int] | tuple[int, int, int, int] = input_nd.shape

        # create output image
        output_nd_height: int = ((d_input_nd[-2] - kernel_size[-2]) // stride[-2]) + 1
        output_nd_width: int = ((d_input_nd[-1] - kernel_size[-1]) // stride[-1]) + 1
        output_nd: ndarray = zeros((*d_input_nd[:-2], output_nd_height, output_nd_width))
        indices_nd: ndarray = zeros((*d_input_nd[:-2], output_nd_height, output_nd_width), dtype=int64)

        # 3D max pooling loop
        i_input_nd: int = 0
        for i_output_nd in range(output_nd_height):
            j_input_nd: int = 0
            for j_output_nd in range(output_nd_width):
                values: ndarray = input_tensor.nd[..., i_input_nd:(i_input_nd + kernel_size[-2]), j_input_nd:(j_input_nd + kernel_size[-1])]
                output_nd[..., i_output_nd, j_output_nd] = amax(values, axis=(-1, -2))
                indices_nd[..., i_output_nd, j_output_nd] = argmax(values.reshape((*d_input_nd[:-2], -1)), axis=-1)
                j_input_nd += stride[-1]
            i_input_nd += stride[-2]
        
        if return_indices:
            return output_nd, indices_nd
        else:
            return output_nd

    output_nd, indices_nd = _maxpool3d(input_tensor.nd, kernel_size, stride, return_indices=True)

    def grad_fn(child: Tensor) -> None:
        d_input_grad: tuple[int, int, int] | tuple[int, int, int, int] = input_tensor.grad.shape
        d_child_grad: tuple[int, int, int] | tuple[int, int, int, int] = child.grad.shape
        i_input_grad: int = 0
        for i_child_grad in range(d_child_grad[-2]):
            j_input_grad: int = 0
            for j_child_grad in range(d_child_grad[-1]):
                if input_tensor.grad.ndim == 3:
                    batches_nd: ndarray = Ellipsis
                    channels_nd: ndarray = arange(d_input_grad[-3])
                else:
                    batches_nd: ndarray = repeat(arange(d_input_grad[-4]), d_input_grad[-3])
                    channels_nd: ndarray = tile(arange(d_input_grad[-3]), d_input_grad[-4])
                print(input_tensor.grad[..., i_input_grad:(i_input_grad + kernel_size[-2]), j_input_grad:(j_input_grad + kernel_size[-1])].reshape((*d_input_grad[:-2], -1))[batches_nd, channels_nd, indices_nd[..., i_child_grad, j_child_grad].flatten()])
                print(child.grad[..., i_child_grad, j_child_grad].flatten())
                input_tensor.grad[..., i_input_grad:(i_input_grad + kernel_size[-2]), j_input_grad:(j_input_grad + kernel_size[-1])].reshape((*d_input_grad[:-2], -1))[batches_nd, channels_nd, indices_nd[..., i_child_grad, j_child_grad].flatten()] += child.grad[..., i_child_grad, j_child_grad].flatten()
                print(input_tensor.grad[..., i_input_grad:(i_input_grad + kernel_size[-2]), j_input_grad:(j_input_grad + kernel_size[-1])].reshape((*d_input_grad[:-2], -1))[batches_nd, channels_nd, indices_nd[..., i_child_grad, j_child_grad].flatten()])
                j_input_grad += stride[-1]
            i_input_grad += stride[-2]

    return Tensor(output_nd, [input_tensor], is_leaf=False, grad_fn=grad_fn)