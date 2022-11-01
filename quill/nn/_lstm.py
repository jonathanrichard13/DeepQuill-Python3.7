from cupy import zeros

from . import Module, Linear
from .functional import add, multiply, sigmoid, split, squeeze, stack, tanh
from ..classes import Tensor
from ..functions import type_check, expr_check

class _Cell(Module):

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        
        # Initialize parent class
        super().__init__()
        
        # TYPE CHECKS
        type_check(input_size, "input_size", int)
        type_check(hidden_size, "output_size", int)
        type_check(bias, "bias", bool)

        # VALUE OUT OF RANGE CHECKS
        expr_check(input_size, "input_size", lambda x: x > 0)
        expr_check(hidden_size, "hidden_size", lambda x: x > 0)

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.bias: bool = bias

        self.U: Linear = Linear(input_size, hidden_size * 4, bias=bias)
        self.W: Linear = Linear(hidden_size, hidden_size * 4, bias=bias)
    
    def forward(self, x: Tensor, hc: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        _h, _c = hc
        x_f, x_i, x_c, x_o = split(self.U(x), 4, axis=-1)
        h_f, h_i, h_c, h_o = split(self.W(_h), 4, axis=-1)
        f: Tensor = sigmoid(add(x_f, h_f))
        i: Tensor = sigmoid(add(x_i, h_i))
        o: Tensor = sigmoid(add(x_o, h_o))
        c_tilde: Tensor = tanh(add(x_c, h_c))
        c: Tensor = add(multiply(f, _c), multiply(i, c_tilde))
        h: Tensor = multiply(o, tanh(c))
        return h, c

class LSTM(Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True) -> None:
        
        # Initialize parent class
        super().__init__()
        
        # TYPE CHECKS
        type_check(input_size, "input_size", int)
        type_check(hidden_size, "output_size", int)
        type_check(num_layers, "num_layers", int)
        type_check(bias, "bias", bool)

        # VALUE OUT OF RANGE CHECKS
        expr_check(input_size, "input_size", lambda x: x > 0)
        expr_check(hidden_size, "hidden_size", lambda x: x > 0)
        expr_check(num_layers, "num_layers", lambda x: x > 0)

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bias: bool = bias

        self.cells: list[_Cell] = [_Cell(input_size, hidden_size, bias)]
        self.cells.extend([_Cell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)])
        for i_cell, cell in enumerate(self.cells):
            setattr(self, f"cell_{i_cell}", cell)

    def forward(self, x: Tensor, hc: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        
        if hc is None:
            hc = (Tensor(zeros((*x.nd.shape[:-2], self.hidden_size))), Tensor(zeros((*x.nd.shape[:-2], self.hidden_size))))
        
        _y: list[Tensor] = []
        for x_t in [squeeze(tensor, -2) for tensor in split(x, x.nd.shape[-2], -2)]:
            for cell in self.cells:
                hc = cell(x_t, hc)
                x_t = hc[0]
            _y.append(x_t)
        
        y: Tensor = stack(_y, axis=-2)
        return y, hc