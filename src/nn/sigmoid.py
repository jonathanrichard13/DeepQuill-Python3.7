from .module import Module
from .functional.sigmoid import sigmoid
from ..classes.tensor import Tensor

class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)