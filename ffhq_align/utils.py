from typing import Sequence, Tuple, Union, cast

from torch import Tensor

INT = Union[int, Sequence[int]]
FLOAT = Union[float, Sequence[float]]


def normalize_int_tuple(value: Union[INT, Tensor], n: int) -> Tuple[int, ...]:
    if isinstance(value, Tensor):
        numel = value.numel()
        if numel != 1:
            raise ValueError(f"Expected a tensor with 1 element, got {numel}")
        value = cast(int, value.item())

    if isinstance(value, int):
        return (value,) * n

    if isinstance(value, map):
        value = tuple(value)

    if not all(isinstance(v, int) for v in value):
        raise TypeError(f"Expected int elements, got {value}.")

    if len(value) == 1 or len(set(value)) == 1:
        return (value[0],) * n

    if len(value) != n:
        raise ValueError(f"The argument must be a tuple of {n} integers, got {value}.")
    return tuple(value)


def normalize_float_tuple(
    value: Union[FLOAT, INT, Tensor], n: int
) -> Tuple[float, ...]:
    if isinstance(value, Tensor):
        numel = value.numel()
        if numel != 1:
            raise ValueError(f"Expected a tensor with 1 element, got {numel}")
        value = value.item()

    if isinstance(value, (int, float)):
        return (float(value),) * n

    if isinstance(value, map):
        value = tuple(value)

    if not all(isinstance(v, (int, float)) for v in value):
        raise TypeError(f"Expected float elements, got {value}.")

    if len(value) == 1 or len(set(value)) == 1:
        return (float(value[0]),) * n

    if len(value) != n:
        raise ValueError(f"The argument must be a tuple of {n} floats, got {value}.")
    return tuple(map(float, value))
