from ..classes import Tensor

class Module:

    def __init__(self) -> None:
        if type(self) is Module:
            raise NotImplementedError("Cannot instantiate abstract class Module. Please extend it to another class first.")
        self.training: bool = True
        
    def train(self) -> None:
        self.training = True
    
    def eval(self) -> None:
        self.training = False
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"Forward function for module type {type(self)} has not been implemented.")

    def __call__(self, x: Tensor) -> Tensor:
        y: Tensor = self.forward(x)
        return y