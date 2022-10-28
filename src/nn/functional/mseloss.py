from cupy import power

from ...classes import Tensor

def mseloss(y_tilde: Tensor, y: Tensor):

    def grad_fn(child: Tensor) -> None:
        y_tilde.grad += child.grad * (y_tilde.nd - y.nd)
        y.grad += child.grad * (y.nd - y_tilde.nd)
    
    return Tensor(power((y_tilde.nd - y.nd), 2) * 0.5, [y_tilde, y], is_leaf=False, grad_fn=grad_fn)