from collections.abc import Callable
from cupy import zeros

from ..core import Tensor

_State = Tensor | dict[str, "_State"]

class Optimizer:

    def __init__(self, params: dict[str, _State]) -> None:
        self.params: dict[str, _State] = params
    
    def _modify_params(self, expr: Callable[[Tensor], None]) -> None:
        values: list[_State] = list(self.params.values())
        visited: list[_State] = []
        while len(values) > 0:
            v: _State = values.pop(0)
            if v not in visited:
                if isinstance(v, dict):
                    values.extend(list(v.values()))
                else:
                    expr(v)
                visited.append(v)
    
    def reset_velocity(self) -> None:
        def _reset_velocity(x: Tensor) -> None:
            if self in x.velocities:
                del x.velocities[self]
        self._modify_params(_reset_velocity)
    
    def clear_velocities(self) -> None:
        def _clear_velocities(x: Tensor) -> None:
            x.velocities.clear()
        self._modify_params(_clear_velocities)
    
    def zero_grad(self) -> None:
        def _zero_grad(x: Tensor) -> None:
            x.i_backward = 0
            x.grad = zeros(x.grad.shape)
        self._modify_params(_zero_grad)
    
    def step_fn(self, x: Tensor) -> None:
        raise NotImplementedError(f"Step function for {type(self).__name__} optimizer has not been implemented yet.")
    
    def step(self) -> None:
        def _step(x: Tensor) -> None:
            x.n_backward = 0
            self.step_fn(x)
        self._modify_params(_step)