from typing import Union
from cupy.random import rand

from . import Module
from .functional import addbias, expand_dims, matmul, squeeze
from ..classes import Tensor
from ..functions import expr_check, type_check

class Linear(Module):

    def __init__(self, input_size: int, output_size: int, bias: bool = True):

        # Initialize parent class
        super().__init__()
        
        # TYPE CHECKS
        type_check(input_size, "input_size", int)
        type_check(output_size, "output_size", int)
        type_check(bias, "bias", bool)

        # VALUE OUT OF RANGE CHECKS
        expr_check(input_size, "input_size", lambda x: x > 0)
        expr_check(output_size, "output_size", lambda x: x > 0)

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.weight: Tensor = Tensor(rand(input_size, output_size))
        self.bias: Union[Tensor, None] = None
        if bias:
            self.bias = Tensor(rand(output_size))

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = squeeze(matmul(expand_dims(x, -2), self.weight), -2)
        if self.bias is not None:
            y = addbias(y, self.bias)
        return y