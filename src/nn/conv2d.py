from cupy.random import rand

from .module import Module
from ..classes.tensor import Tensor
from ..functional.addconvbias import addconvbias
from ..functional.conv3d import conv3d
from ..functional.sum import sum
from ..functional.stack import stack
from ..functional.unstack import unstack

class Conv2d(Module):

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, bias: bool = True):
        
         # TYPE CHECKS
        if not isinstance(input_channels, int):
            raise TypeError(f"Parameter input_channels ({input_channels}) must be an integer.")
        if not isinstance(output_channels, int):
            raise TypeError(f"Parameter output_channels ({output_channels}) must be an integer.")
        if (not isinstance(kernel_size, int)) and (not isinstance(kernel_size, tuple)):
            raise TypeError(f"Parameter kernel_size ({kernel_size}) must be either an integer or a tuple.")
        if (not isinstance(stride, int)) and (not isinstance(stride, tuple)):
            raise TypeError(f"Parameter stride ({stride}) must be either an integer or a tuple.")
        if (not isinstance(padding, int)) and (not isinstance(padding, tuple)):
            raise TypeError(f"Parameter padding ({padding}) must be either an integer or a tuple.")
        if (not isinstance(bias, bool)):
            raise TypeError(f"Parameter bias ({bias}) must be a boolean value.")
    
        # cast kernel_size, stride, padding into tuple
        if isinstance(kernel_size, int):
            kernel_size: tuple[int, int] = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride: tuple[int, int] = (stride, stride)
        if isinstance(padding, int):
            padding: tuple[int, int] = (padding, padding)

        # MISMATCHED DIMENSION CHECKS
        if len(kernel_size) != 2:
            raise IndexError(f"If kernel_size ({kernel_size}) is a tuple, it must be of length = 2.")
        if len(stride) != 2:
            raise IndexError(f"If stride ({stride}) is a tuple, it must be of length = 2.")
        if len(padding) != 2:
            raise IndexError(f"If padding ({padding}) is a tuple, it must be of length = 2.")

        self.weight: Tensor = Tensor(rand(output_channels, input_channels, kernel_size[0], kernel_size[1]))
        self.bias: Tensor | None = None
        if bias:
            self.bias = Tensor(rand(output_channels))
        self.stride: int | tuple[int, int] = stride
        self.padding: int | tuple[int, int] = padding

    def forward(self, x: Tensor) -> Tensor:
        y_unstacked: list[Tensor] = []
        weights: list[Tensor] = unstack(self.weight)
        for weight in weights:
            y_unstacked.append(sum(conv3d(x, weight, self.stride, self.padding), axis=-3))
        y: Tensor = stack(y_unstacked, axis=-3)
        if self.bias is not None:
            y = addconvbias(y, self.bias)
        return y