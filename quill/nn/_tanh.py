from . import Module
from .functional import tanh
from ..classes import Tensor

class Tanh(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)