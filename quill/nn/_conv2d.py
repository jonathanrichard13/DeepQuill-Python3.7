from collections.abc import Sequence
from cupy.random import rand

from . import Module
from .functional import addbias, conv3d, sum, stack, unstack
from ..classes import Tensor
from ..functions import expr_check, len_check, type_check

class Conv2d(Module):

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int | Sequence[int], stride: int | Sequence[int] = 1, padding: int | Sequence[int] = 0, bias: bool = True):

        # Initialize parent class
        super().__init__()
        
        # TYPE CHECKS
        type_check(input_channels, "input_channels", int)
        type_check(output_channels, "output_channels", int)
        type_check(kernel_size, "kernel_size", (int | Sequence), int)
        type_check(stride, "stride", (int | Sequence), int)
        type_check(padding, "padding", (int | Sequence), int)
        type_check(bias, "bias", bool)
    
        # cast kernel_size, stride, padding into tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        # MISMATCHED DIMENSION CHECKS
        len_check(kernel_size, "kernel_size", 2)
        len_check(stride, "stride", 2)
        len_check(padding, "padding", 2)

        # VALUE OUT OF RANGE CHECKS
        expr_check(input_channels, "input_channels", lambda x: x > 0)
        expr_check(output_channels, "output_channels", lambda x: x > 0)
        for i_kernel_size in range(len(kernel_size)):
            expr_check(kernel_size[i_kernel_size], f"kernel_size[{i_kernel_size}]", lambda x: x > 0)
        for i_stride in range(len(stride)):
            expr_check(stride[i_stride], f"stride[{i_stride}]", lambda x: x > 0)
        for i_padding in range(len(padding)):
            expr_check(padding[i_padding], f"padding[{i_padding}]", lambda x: x >= 0)

        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.kernel_size: tuple[int, int] = kernel_size
        self.stride: tuple[int, int] = stride
        self.padding: tuple[int, int] = padding
        
        self.weight: Tensor = Tensor(rand(output_channels, input_channels, kernel_size[0], kernel_size[1]))
        self.bias: Tensor | None = None
        if bias:
            self.bias = Tensor(rand(output_channels))

    def forward(self, x: Tensor) -> Tensor:
        _y: list[Tensor] = []
        for w in unstack(self.weight, axis=0):
            _y.append(sum(conv3d(x, w, self.stride, self.padding), axis=-3))
        y: Tensor = stack(_y, axis=-3)
        if self.bias is not None:
            y = addbias(y, self.bias, axis=-3)
        return y