from cupy import arange, ones
from src.classes import Tensor
from src.nn import Conv2d

# Unbatched

x: Tensor = Tensor(arange(48).reshape((3, 4, 4)))
f: Conv2d = Conv2d(3, 2, 2)
f.weight.nd = ones((2, 3, 2, 2))
f.bias.nd = arange(2)
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

x: Tensor = Tensor(arange(240).reshape((5, 3, 4, 4)))
f: Conv2d = Conv2d(3, 2, 2)
f.weight.nd = ones((2, 3, 2, 2))
f.bias.nd = arange(2)
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