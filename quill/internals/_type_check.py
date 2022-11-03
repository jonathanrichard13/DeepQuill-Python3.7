from collections.abc import Collection
from typing import Any

def type_check(obj: Any, nameof_obj: str, classinfo: Any, parameterized_generic_classinfo: Any | None = None) -> None:
    if not isinstance(obj, classinfo):
        raise ValueError(f"`{nameof_obj}` must be of type `{classinfo.__name__}`, it cannot be of type `{type(obj).__name__}`.")
    elif (parameterized_generic_classinfo is not None) and (isinstance(obj, Collection)):
        for el in obj:
            if not isinstance(el, parameterized_generic_classinfo):
                raise ValueError(f"{nameof_obj}` must be of type `{classinfo.__name__}[{parameterized_generic_classinfo.__name__}]`, it cannot contain objects of type `{type(el).__name__}`.")