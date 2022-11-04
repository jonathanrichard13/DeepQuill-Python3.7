from typing import Dict, Union

from . import Optimizer
from ..core import Tensor

_State = Union[Tensor, Dict[str, "_State"]]

class SGD(Optimizer):

    def __init__(self, params: Dict[str, _State], lr: float = 0.001, momentum: float = 0.):
        super().__init__(params)
        self.lr: float = lr
        self.momentum: float = momentum
    
    def step_fn(self, x: Tensor) -> None:
        if self.momentum != 0:
            if self not in x.velocities:
                x.velocities[self] = self.lr * x.grad
            else:
                x.velocities[self] = (self.momentum * x.velocities[self]) + (self.lr * x.grad)
            x.nd -= x.velocities[self]
        else:
            x.nd -= self.lr * x.grad