from numpy import mean, ndarray, reciprocal, zeros

from ..classes.tensor import Tensor

def avgpool2d(input_tensor: Tensor, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> Tensor:

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
    if input_tensor.nd.ndim != 2:
        raise IndexError(f"Input tensor is {input_tensor.nd.ndim}-dimensional instead of 2-dimensional.")
    if len(kernel_size) != 2:
        raise IndexError(f"If kernel_size ({kernel_size}) is a tuple, it must be of length = 2.")
    if len(stride) != 2:
        raise IndexError(f"If stride ({stride}) is a tuple, it must be of length = 2.")
    
    def _avgpool2d(input_nd: ndarray, kernel_size: tuple[int, int], stride: tuple[int, int]) -> ndarray:

        # get input dimensions
        d_input_nd = input_nd.shape

        # create output image
        output_nd_height: int = ((d_input_nd[0] - kernel_size[0]) // stride[0]) + 1
        output_nd_width: int = ((d_input_nd[1] - kernel_size[1]) // stride[1]) + 1
        output_nd: ndarray = zeros((output_nd_height, output_nd_width))

        # 2D average pooling loop
        i_input_nd: int = 0
        for i_output_nd in range(output_nd_height):
            j_input_nd: int = 0
            for j_output_nd in range(output_nd_width):
                output_nd[i_output_nd, j_output_nd] = mean(input_nd[i_input_nd:(i_input_nd + kernel_size[0]), j_input_nd:(j_input_nd + kernel_size[1])])
                j_input_nd += stride[1]
            i_input_nd += stride[0]
        
        return output_nd

    def grad_fn(child: Tensor) -> None:
        i_input_tensor_grad: int = 0
        for row in child.grad:
            j_input_tensor_grad: int = 0
            for value in row:
                input_tensor.grad[i_input_tensor_grad:(i_input_tensor_grad + kernel_size[0]), j_input_tensor_grad:(j_input_tensor_grad + kernel_size[1])] += value * reciprocal(kernel_size[0] * kernel_size[1])
                j_input_tensor_grad += stride[1]
            i_input_tensor_grad += stride[0]

    return Tensor(_avgpool2d(input_tensor.nd, kernel_size, stride), [input_tensor], is_leaf=False, grad_fn=grad_fn)