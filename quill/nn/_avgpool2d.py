from typing import Sequence, Tuple, Union

from . import Module
from .functional import avgpool3d
from ..classes import Tensor
from ..functions import type_check, len_check, expr_check

class AvgPool2d(Module):

    def __init__(self, kernel_size: Union[int, Sequence[int]], stride: Union[int, Sequence[int], None] = None) -> None:

        # Initialize parent class
        super().__init__()

        # TYPE CHECKS
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
        len_check(kernel_size, "kernel_size", 2)
        len_check(stride, "stride", 2)

        # VALUE OUT OF RANGE CHECKS
        for i_kernel_size in range(len(kernel_size)):
            expr_check(kernel_size[i_kernel_size], f"kernel_size[{i_kernel_size}]", lambda x: x > 0)
        for i_stride in range(len(stride)):
            expr_check(stride[i_stride], f"stride[{i_stride}]", lambda x: x > 0)

        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: Tuple[int, int] = stride
    
    def forward(self, x: Tensor) -> Tensor:
        return avgpool3d(x, self.kernel_size, self.stride)