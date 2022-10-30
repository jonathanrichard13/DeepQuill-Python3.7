from typing import Callable
from cupy import zeros

from ..classes import Tensor

_State = Tensor | dict[str, "_State"]

class Optimizer:

    def __init__(self, params: dict[str, _State]) -> None:
        self.params: dict[str, _State] = params
    
    def _modify_params(self, expr: Callable[[Tensor], None]) -> None:
        _params: list[_State] = list(self.params.values())
        visited: set[_State] = set()
        while len(_params) > 0:
            v = _params.pop()
            if isinstance(v, dict):
                if v not in visited:
                    _params.extend(list(v.values()))
            else:
                expr(v)
            visited.add(v)
    
    def zero_grad(self) -> None:
        def _zero_grad_and_reset_velocity(x: Tensor) -> None:
            x.grad = zeros(x.grad.shape)
            x.velocity = None
        self._modify_params(_zero_grad_and_reset_velocity)
    
    def step(self) -> None:
        raise NotImplementedError(f"Step function for {type(self).__name__} optimizer has not been implemented yet.")