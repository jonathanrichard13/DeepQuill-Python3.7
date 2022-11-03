from typing import Callable, Dict, List, Union
from cupy import zeros

from ..core import Tensor

_State = Union[Tensor, Dict[str, "_State"]]

class Optimizer:

    def __init__(self, params: Dict[str, _State]) -> None:
        self.params: Dict[str, _State] = params
    
    def _modify_params(self, expr: Callable[[Tensor], None]) -> None:
        values: List[_State] = list(self.params.values())
        visited: List[_State] = []
        while len(values) > 0:
            v: _State = values.pop(0)
            if v not in visited:
                if isinstance(v, dict):
                    values.extend(list(v.values()))
                else:
                    expr(v)
                visited.append(v)
    
    def zero_grad(self) -> None:
        def _zero_grad_and_reset_velocity(x: Tensor) -> None:
            x.grad = zeros(x.grad.shape)
            x.velocity = None
        self._modify_params(_zero_grad_and_reset_velocity)
    
    def step(self) -> None:
        raise NotImplementedError(f"Step function for {type(self).__name__} optimizer has not been implemented yet.")