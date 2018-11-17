import numpy as np


def abs_core(p, m, is_diff=False):
    if is_diff:
        def f(e):
            return np.sign(e)
    else:
        def f(e):
            return p * abs(e)
    return f


def abs_pow_core(p, m, is_diff=False):
    if is_diff:
        def f(e):
            return np.power(abs(e), m - 1)
    else:
        def f(e):
            return p * np.power(abs(e), 2 - m)
    return f


def piecewise_core(p, m, is_diff=False):
    if is_diff:
        def f(e):
            return e if abs(e) <= m else m * np.sign(e)
    else:
        def f(e):
            return p if abs(e) <= m else p * abs(e) / m
    return f


cores_dict = {  # 6.7.2 (функция, производная)
    'abs': abs_core,
    'abs_pow': abs_pow_core,
    'piecewise': piecewise_core
}