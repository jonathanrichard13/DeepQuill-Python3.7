from . import Module
from .functional import maxpool3d
from ..classes import Tensor

class MaxPool2d(Module):

    def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> None:
        super().__init__()
        self.kernel_size: int | tuple[int, int] = kernel_size
        self.stride: int | tuple[int, int] | None = stride
    
    def forward(self, x: Tensor) -> Tensor:
        return maxpool3d(x, self.kernel_size, self.stride)