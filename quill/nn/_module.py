from typing import Any, Dict, Union

from ..classes import Tensor

_State = Union[Tensor, Dict[str, "_State"]]

class Module:

    def __init__(self) -> None:
        self.training: bool = True
        
    def train(self) -> None:
        self.training = True
    
    def eval(self) -> None:
        self.training = False
    
    def forward(self, x: Tensor, *args) -> Tensor:
        raise NotImplementedError(f"Forward function for module type {type(self).__name__} has not been implemented yet.")

    def parameters(self) -> Dict[str, _State]:
        state_dict: Dict[str, _State] = {}
        for k, v in vars(self).items():
            if isinstance(v, Tensor):
                state_dict[k] = v
            elif isinstance(v, Module):
                state_dict[k] = v.parameters()
        return state_dict
    
    def __call__(self, x: Tensor, *args) -> Tensor:
        return self.forward(x, *args)

    def __repr__(self) -> str:
        lines: str = f"{type(self).__name__}("
        default_attributes: Dict[str, Any] = vars(Module())
        parameters: Dict[str, Any] = {}
        submodules: Dict[str, Any] = {}
        for k, v in vars(self).items():
            if k not in default_attributes:
                if isinstance(v, Module):
                    submodules[k] = "\n  ".join(str(v).split("\n"))
                elif k == "bias":
                    if isinstance(v, bool):
                        parameters[k] = v
                    elif isinstance(v, Tensor):
                        parameters[k] = True
                    elif v is None:
                        parameters[k] = False
                elif not isinstance(v, Tensor):
                    parameters[k] = v
        if len(submodules) == 0:
            lines += ", ".join([f"{k}={v}" for k, v in parameters.items()]) + ")"
        else:
            lines += "".join([f"\n  ({k}): {v}" for k, v in submodules.items()]) + "\n)"
        return lines