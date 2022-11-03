from typing import Any, Collection, Tuple, Union

def type_check(obj: Any, nameof_obj: str, classinfo: Union[Any, Tuple[Any, ...]], parameterized_generic_classinfo: Union[Any, None] = None) -> None:
    if not isinstance(obj, classinfo):
        if isinstance(classinfo, Collection):
            raise ValueError(f"`{nameof_obj}` must be one of these types: `{classinfo}`, it cannot be of type `{type(obj).__name__}`.")
        else:
            raise ValueError(f"`{nameof_obj}` must be of type `{classinfo.__name__}`, it cannot be of type `{type(obj).__name__}`.")
    elif (parameterized_generic_classinfo is not None) and (isinstance(obj, Collection)):
        for el in obj:
            if not isinstance(el, parameterized_generic_classinfo):
                if isinstance(classinfo, Collection):
                    raise ValueError(f"{nameof_obj}` must be one of these types `{classinfo}[{parameterized_generic_classinfo.__name__}]`, it cannot contain objects of type `{type(el).__name__}`.")
                else:    
                    raise ValueError(f"{nameof_obj}` must be of type `{classinfo.__name__}[{parameterized_generic_classinfo.__name__}]`, it cannot contain objects of type `{type(el).__name__}`.")