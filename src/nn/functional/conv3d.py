from numpy import ndarray
from cupy import flip, multiply, pad, sum, zeros

from ...classes import Tensor

def conv3d(input_tensor: Tensor, kernel: Tensor, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0) -> Tensor:
    
    # TYPE CHECKS
    if not isinstance(input_tensor, Tensor):
        raise TypeError(f"Parameter input_tensor ({input_tensor}) must be a Tensor.")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Parameter kernel ({kernel}) must be a Tensor.")
    if (not isinstance(stride, int)) and (not isinstance(stride, tuple)):
        raise TypeError(f"Parameter stride ({stride}) must be either an integer or a tuple.")
    if (not isinstance(padding, int)) and (not isinstance(padding, tuple)):
        raise TypeError(f"Parameter padding ({padding}) must be either an integer or a tuple.")

    # cast stride, padding, and dilation into tuple
    if isinstance(stride, int):
        stride: tuple[int, int] = (stride, stride)
    if isinstance(padding, int):
        padding: tuple[int, int] = (padding, padding)
    
    # get ndarrays
    input_nd: ndarray = input_tensor.nd
    kernel_nd: ndarray = kernel.nd

    # MISMATCHED DIMENSION CHECKS
    if (input_nd.ndim != 3) and (input_nd.ndim != 4):
        raise IndexError(f"Input tensor is {input_nd.ndim}-dimensional instead of 3-or-4-dimensional.")
    if kernel_nd.ndim != 3:
        raise IndexError(f"Kernel is {kernel_nd.ndim}-dimensional instead of 3-dimensional.")
    if input_nd.shape[-3] != kernel_nd.shape[-3]:
        raise IndexError(f"Input tensor with size {input_nd.shape} cannot be convolved with kernel of size {kernel_nd.shape}.")
    if len(stride) != 2:
        raise IndexError(f"If stride ({stride}) is a tuple, it must be of length = 2.")
    if len(padding) != 2:
        raise IndexError(f"If padding ({padding}) is a tuple, it must be of length = 2.")

    # VALUE OUT OF RANGE CHECKS
    for value in stride:
        if value < 1:
            raise ValueError(f"Stride must be positive, not {stride}.")
    for value in padding:
        if value < 0:
            raise ValueError(f"Padding must be non-negative, not {padding}.")

    # conv3d function that returns an ndarray instead of Tensor (if numpy has a 3D convolution function, we would've borrowed it instead)
    def _conv3d(x1: ndarray, x2: ndarray, stride: tuple[int, int], padding: tuple[int, int]) -> ndarray:

        # pad input image
        for p in padding:
            if p != 0:
                if x1.ndim == 3:
                    x1 = pad(x1, ((0, 0), (padding[-2], padding[-2]), (padding[-1], padding[-1])))
                else:
                    x1 = pad(x1, ((0, 0), (0, 0), (padding[-2], padding[-2]), (padding[-1], padding[-1])))
                break

        # get input and kernel dimensions
        d_x1: tuple[int, int, int] | tuple[int, int, int, int] = x1.shape
        d_x2: tuple[int, int, int] = x2.shape
        
        # create output image
        y_height: int = ((d_x1[-2] + 2 * padding[-2] - d_x2[-2]) // stride[-2]) + 1
        y_width: int = ((d_x1[-1] + 2 * padding[-1] - d_x2[-1]) // stride[-1]) + 1
        y: ndarray = zeros((*d_x1[:-2], y_height, y_width))

        # 3D cross-correlation loop
        for i_y, i_x1 in zip(range(y_height), range(0, (d_x1[-2] - d_x2[-2] + 1), stride[-2])):
            for j_y, j_x1 in zip(range(y_width), range(0, (d_x1[-1] - d_x2[-1] + 1), stride[-1])):
                y[..., i_y, j_y] = sum(multiply(x1[..., i_x1:(i_x1 + d_x2[-2]), j_x1:(j_x1 + d_x2[-1])], x2), axis=(-1, -2))
                
        return y

    # dilation function, only used by grad_fn
    def _dilate(x: ndarray, dilation: tuple[int, int]) -> ndarray:
        d_x: tuple[int, int, int] | tuple[int, int, int, int] = x.shape
        y_height: int = d_x[-2] + ((d_x[-2] - 1) * (dilation[-2] - 1))
        y_width: int = d_x[-1] + ((d_x[-1] - 1) * (dilation[-1] - 1))
        y: ndarray = zeros((*d_x[:-2], y_height, y_width))
        for i_x, i_y in zip(range(d_x[-2]), range(0, y_height, dilation[-2])):
            for j_x, j_y in zip(range(d_x[-1]), range(0, y_width, dilation[-1])):
                y[..., i_y, j_y] = x[..., i_x, j_x]
        return y
            
    def grad_fn(child: Tensor) -> None:

        # dilate grad according to stride
        grad: ndarray = _dilate(child.grad, stride)

        # calculate 'outer padding' (pad for pixels that were not convoluted in forward pass)
        d_input: tuple[int, int, int] | tuple[int, int, int, int] = input_nd.shape
        d_kernel: tuple[int, int, int] = kernel_nd.shape
        outer_padding_height: int = (d_input[-2] + 2 * padding[-2] - d_kernel[-2]) % stride[-2]
        outer_padding_width: int = (d_input[-1] + 2 * padding[-1] - d_kernel[-1]) % stride[-1]
        
        # calculate gradient for input image
        _grad: ndarray = _conv3d(grad, flip(kernel_nd, axis=(-1, -2)), (1, 1), (d_kernel[-2] - 1, d_kernel[-1] - 1))
        if _grad.ndim == 3:
            _grad = pad(_grad, ((0, 0), (0, outer_padding_height), (0, outer_padding_width)))
        else:
            _grad = pad(_grad, ((0, 0), (0, 0), (0, outer_padding_height), (0, outer_padding_width)))
        input_tensor.grad += _grad[..., padding[-2]:(padding[-2] + d_input[-2]), padding[-1]:(padding[-1] + d_input[-1])]

        # calculate gradient for kernel
        d_grad: tuple[int, int, int] | tuple[int, int, int, int] = grad.shape
        _grad: ndarray = _conv3d(input_nd.reshape((-1, d_input[-2], d_input[-1])), grad.reshape((-1, d_grad[-2], d_grad[-1])), (1, 1), padding)[:, :d_kernel[-2], :d_kernel[-1]]
        n_channel: int = kernel.grad.shape[-3]
        for i_channel in range(n_channel):
            kernel.grad[i_channel, :, :] += sum(_grad[i_channel::n_channel, :, :], axis=0)

    return Tensor(_conv3d(input_nd, kernel_nd, stride, padding), [input_tensor, kernel], is_leaf=False, grad_fn=grad_fn)