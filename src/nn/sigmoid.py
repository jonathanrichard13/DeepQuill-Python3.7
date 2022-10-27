from .module import Module
from ..classes.tensor import Tensor
from ..functional.sigmoid import sigmoid

class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)