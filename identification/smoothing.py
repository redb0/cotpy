from typing import Union, Tuple


Number = Union[int, float]


def average(a: Number, last_a: Number, n: int) -> Number:
    # TODO: документация
    return last_a + (1 / n) * (a - last_a)


def efi(a: Number, last_a: Number, l: Number, last_sigma: Number) -> Tuple[Number, Number]:
    # TODO: документация
    sigma = 1 + l * last_sigma
    new_a = last_a + (1 / sigma) * (a - last_a)
    return new_a, sigma


def moving_average(a: Number, last_a: Number, last_k_a: Number, k: int) -> Number:
    # TODO: документация
    return last_a + (1 / k) * (a - last_k_a)
