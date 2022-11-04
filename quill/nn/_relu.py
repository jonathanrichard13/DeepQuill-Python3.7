from . import Module
from .functional import relu
from ..core import Tensor

class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)