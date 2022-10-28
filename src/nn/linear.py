from cupy.random import rand

from . import Module
from .functional import addbias, expand_dims, matmul, squeeze
from ..classes import Tensor

class Linear(Module):

    def __init__(self, input_size: int, output_size: int, bias: bool = True):

        # Initialize parent class
        super().__init__()
        
         # TYPE CHECKS
        if not isinstance(input_size, int):
            raise TypeError(f"Parameter input_size ({input_size}) must be an integer.")
        if not isinstance(output_size, int):
            raise TypeError(f"Parameter output_size ({output_size}) must be an integer.")
        if (not isinstance(bias, bool)):
            raise TypeError(f"Parameter bias ({bias}) must be a boolean value.")

        self.weight: Tensor = Tensor(rand(input_size, output_size))
        self.bias: Tensor | None = None
        if bias:
            self.bias = Tensor(rand(output_size))

    def forward(self, x: Tensor) -> Tensor:
        return addbias(squeeze(matmul(expand_dims(x, -2), self.weight), -2), self.bias)