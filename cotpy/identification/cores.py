"""
Модуль функций ядер для квадратичных критериев.
:Authors:
    - Vladimir Voronov
"""


import numpy as np


def abs_core(p, m, is_diff=False):
    """
    Модульное ядро.
    :param p      : коэффициент крутости
    :type p       : number
    :param m      : -
    :param is_diff: флаг производной, если True - считается производная ядра.
    :type is_diff : bool
    :return: function(e:number) -> number
    """
    if is_diff:
        def f(e):
            return np.sign(e)
    else:
        def f(e):
            return p * abs(e)
    return f


def abs_pow_core(p, m, is_diff=False):
    """
    Модульное степянное ядро.
    :param p      : коэффициент крутости
    :type p       : number
    :param m      : степень
    :type m       : number
    :param is_diff: флаг производной, если True - считается производная ядра.
    :type is_diff : bool
    :return: function(e:number) -> number
    """
    if is_diff:
        def f(e):
            return np.power(abs(e), m - 1)
    else:
        def f(e):
            return p * np.power(abs(e), 2 - m)
    return f


def piecewise_core(p, m, is_diff=False):
    """
    Кусочное ядро.
    :param p      : коэффициент крутости
    :type p       : number
    :param m      : ширина основы ядра
    :type m       : number
    :param is_diff: флаг производной, если True - считается производная ядра.
    :type is_diff : bool
    :return: function(e:number) -> number
    """
    if is_diff:
        def f(e):
            return e if abs(e) <= m else m * np.sign(e)
    else:
        def f(e):
            return p if abs(e) <= m else p * abs(e) / m
    return f


def power_core(p, s, e_max=1, w_min=0):  # x
    def f(x):
        if abs(x) <= e_max:
            return ((1 - (abs(x) / e_max) ** p) ** s) * (1-w_min) + w_min
        else:
            return w_min
    return f


cores_dict = {  # 6.7.2 (функция, производная)
    'abs': abs_core,
    'abs_pow': abs_pow_core,
    'piecewise': piecewise_core
}