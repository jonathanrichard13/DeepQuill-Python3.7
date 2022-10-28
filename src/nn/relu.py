from .module import Module
from .functional.relu import relu
from ..classes.tensor import Tensor

class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)