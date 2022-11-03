from . import Module
from .functional import sigmoid
from ..core import Tensor

class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)