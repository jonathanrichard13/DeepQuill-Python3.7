from ...classes import Tensor

def mseloss(y_tilde: Tensor, y: Tensor):

    def grad_fn(child: Tensor) -> None:
        y_tilde.grad += (y_tilde.nd - y.nd) * child.grad
        y.grad += (y.nd - y_tilde.nd) * child.grad
    
    return Tensor(((y_tilde.nd - y.nd) ** 2) * 0.5, [y_tilde, y], grad_fn=grad_fn)