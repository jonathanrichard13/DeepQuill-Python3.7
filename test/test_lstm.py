from cupy import arange
from src.classes import Tensor
from src.nn import LSTM
from datetime import datetime

f: LSTM = LSTM(5, 10)

# Unbatched

for i in range(1, 5 + 1):
    start_time = datetime.now()
    x: Tensor = Tensor(arange(i * 5).reshape((i, 5)))
    y, (h, c) = f(x)
    h.backward()
    print(f"i: {i}, unbatched, Time: {datetime.now() - start_time}")

# Batched

    start_time = datetime.now()
    x = Tensor(arange(32 * i * 5).reshape((32, i, 5)))
    y, (h, c) = f(x)
    h.backward()
    print(f"i: {i}, batched, Time: {datetime.now() - start_time}")