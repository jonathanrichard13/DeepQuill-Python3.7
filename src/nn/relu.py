from .module import Module
from ..classes.tensor import Tensor
from ..functional.relu import relu

class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)