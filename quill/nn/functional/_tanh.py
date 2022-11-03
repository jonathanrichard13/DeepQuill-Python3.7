from cupy import exp
from numpy import ndarray

from ...classes import Tensor
from ...internals import type_check

def tanh(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    type_check(x, "x", Tensor)

    def _tanh(x: ndarray) -> ndarray:
        y1: ndarray = exp(x)
        y2: ndarray = exp(-x)
        return (y1 - y2) / (y1 + y2)

    def grad_fn(child: Tensor) -> None:
        x.grad += (1 - (child.nd ** 2)) * child.grad

    return Tensor(_tanh(x.nd), [x], grad_fn=grad_fn)