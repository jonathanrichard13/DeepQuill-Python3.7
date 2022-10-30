from . import Module
from ..classes import Tensor

class Sequential(Module):

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules: list[Module] = []
        for module in modules:
            self.append(module)
    
    def append(self, module: Module) -> None:
        setattr(self, f"{len(self.modules)}", module)
        self.modules.append(module)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = x
        for module in self.modules:
            y = module(y)
        return y