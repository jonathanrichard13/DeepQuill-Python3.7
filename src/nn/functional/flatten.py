from ...classes import Tensor

def flatten(x: Tensor) -> Tensor:

    # TYPE CHECKS
    # x must be a Tensor
    if not isinstance(x, Tensor):
        raise TypeError(f"{x} is not a Tensor.")

    def grad_fn(child: Tensor) -> None:
        x.grad += child.grad.reshape(x.grad.shape)

    return Tensor(x.nd.flatten(), [x], is_leaf=False, grad_fn=grad_fn)