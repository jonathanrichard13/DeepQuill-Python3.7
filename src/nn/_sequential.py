from . import Module
from ..classes import Tensor

class Sequential(Module):

    def __init__(self) -> None:
        super().__init__()
        self.modules: list[Module] = []
    
    def append(self, module: Module) -> None:
        self.modules.append(module)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = x
        for module in self.modules:
            y = module(y)
        return y