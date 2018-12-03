"""
Модуль функций сглаживания.
:Authors:
    - Vladimir Voronov
"""


from typing import Union, Tuple


Number = Union[int, float]


def average(a: Number, last_a: Number, n: int) -> Number:
    """
    Простое сглаживание, аналогичное используется в алгоритме Поляка.
    :param a     : текущее значение коэффициента
    :type a      : int or float
    :param last_a: предыдущее значение коэффициента
    :type last_a : int or float
    :param n     : номер измерения/итерации
    :type n      : int
    :return: значение коэффициента после сглаживания
    :rtype : int or float
    """
    return last_a + (1 / n) * (a - last_a)


def efi(a: Number, last_a: Number, l: Number, last_sigma: Number) -> Tuple[Number, Number]:
    """
    Экспоненциальное забывание информации.
    :param a         : текущее значение коэффициента
    :type a          : int or float
    :param last_a    : предыдущее значение коэффициента
    :type last_a     : int or float
    :param l         : коэффициент забывания информации, 0 < l <=1, 
                       при l = 1 - забывания информации нет. 
                       Рекомендуемое значение 0.9 <= l <= 0.995
    :type l          : int or float
    :param last_sigma: предыдущее значение коэффициента сигма. 
                       Рекомендуемое стартовое значение = 0.
    :type last_sigma : int or float
    :return: кортеж из нового значения коэффициента и нового значения коэффициента сигма.
    :rtype : Tuple[number, number]
    """
    sigma = 1 + l * last_sigma
    new_a = last_a + (1 / sigma) * (a - last_a)
    return new_a, sigma


def moving_average(a: Number, last_a: Number, last_k_a: Number, k: int) -> Number:
    """
    Сглаживание методом скользящего среднего.
    :param a       : текущее значение коэффициента
    :type a        : int or float
    :param last_a  : предыдущее значение коэффициента
    :type last_a   : int or float
    :param last_k_a: значение коэффициента на k-й итерации назад
    :type last_k_a : int or float
    :param k       : количество усредняемых значений
    :type k        : int
    :return: значение коэффициента после сглаживания
    :rtype : int or float
    """
    return last_a + (1 / k) * (a - last_k_a)


average_type = {
    'standard': average,
    'efi': efi,
    'moving': moving_average,
}
