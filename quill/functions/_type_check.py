from typing import Any, Collection, Tuple, Union

def type_check(obj: Any, nameof_obj: str, classinfo: Union[Any, Tuple[Any, ...]], parameterized_generic_classinfo: Union[Any, None] = None) -> None:
    def _type_check(obj: Any, classinfo: Any, parameterized_generic_classinfo: Union[Any, None] = None) -> int:
        if not isinstance(obj, classinfo):
            return -1
        if (parameterized_generic_classinfo is not None) and (isinstance(obj, Collection)):
            for el in obj:
                if not isinstance(el, parameterized_generic_classinfo):
                    return -2
        return 0
    result: bool = False
    if isinstance(classinfo, Collection):
        for _classinfo in classinfo:
            result = _type_check(obj, _classinfo, parameterized_generic_classinfo=parameterized_generic_classinfo)
            if result == 0:
                break
            elif result == -2:
                raise ValueError(f"{nameof_obj}` must be of type `{_classinfo.__name__}[{parameterized_generic_classinfo.__name__}]`, it cannot contain objects of types other than `{parameterized_generic_classinfo.__name__}`.")
        raise ValueError(f"`{nameof_obj}` must be of type `{classinfo.__name__}`, it cannot be of type `{type(obj).__name__}`.")
    else:
        result = _type_check(obj, classinfo, parameterized_generic_classinfo=parameterized_generic_classinfo)
        if result == -1:
            raise ValueError(f"`{nameof_obj}` must be of type `{classinfo.__name__}`, it cannot be of type `{type(obj).__name__}`.")
        elif result == -2:
            raise ValueError(f"{nameof_obj}` must be of type `{classinfo.__name__}[{parameterized_generic_classinfo.__name__}]`, it cannot contain objects of types other than `{parameterized_generic_classinfo.__name__}`.")