from cupy import arange
from src.classes import Tensor
from src.nn import AvgPool2d

# Unbatched

x: Tensor = Tensor(arange(48).reshape((3, 4, 4)))
f: AvgPool2d = AvgPool2d(2)
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

x: Tensor = Tensor(arange(96).reshape((2, 3, 4, 4)))
f: AvgPool2d = AvgPool2d(2)
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