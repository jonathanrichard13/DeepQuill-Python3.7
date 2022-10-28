from collections.abc import Callable
from inspect import getsource
from typing import TypeVar

T = TypeVar("T")

def expr_check(obj: T, nameof_obj: str, expr: Callable[[T], bool]) -> None:
    if not expr(obj):
        raise ValueError(f"`{nameof_obj}` does not satisfy the following expression:\n{getsource(expr)}")