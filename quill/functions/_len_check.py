from typing import Any

def len_check(obj: Any, nameof_obj: str, length: int) -> None:
    if len(obj) != length:
        raise IndexError(f"If `{nameof_obj}` is a `{type(obj).__name__}`, it must be of length = {length}.")