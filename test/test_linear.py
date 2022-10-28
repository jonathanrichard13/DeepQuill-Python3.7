from cupy import arange, ones
from src.classes import Tensor
from src.nn import Linear

# Unbatched

x: Tensor = Tensor(arange(5))
f: Linear = Linear(5, 10)
f.weight.nd = ones((5, 10))
f.bias.nd = arange(10)
y: Tensor = f(x)
y.backward()

print(f"Input -- {x.nd.shape}")
print(x.nd)
print(f"Output -- {y.nd.shape}")
print(y.nd)
print(f"Output grad -- {y.grad.shape}")
print(y.grad)
print(f"Input grad -- {x.grad.shape}")
print(x.grad)

# Batched

x: Tensor = Tensor(arange(10).reshape((2, 5)))
f: Linear = Linear(5, 10)
f.weight.nd = ones((5, 10))
f.bias.nd = arange(10)
y: Tensor = f(x)
y.backward()

print(f"Input -- {x.nd.shape}")
print(x.nd)
print(f"Output -- {y.nd.shape}")
print(y.nd)
print(f"Output grad -- {y.grad.shape}")
print(y.grad)
print(f"Input grad -- {x.grad.shape}")
print(x.grad)