from .module import Module
from ..classes.tensor import Tensor
from ..functional.avgpool3d import avgpool3d

class AvgPool2d(Module):

    def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> None:
        super().__init__()
        self.kernel_size: int | tuple[int, int] = kernel_size
        self.stride: int | tuple[int, int] | None = stride
    
    def forward(self, x: Tensor) -> Tensor:
        return avgpool3d(x, self.kernel_size, self.stride)