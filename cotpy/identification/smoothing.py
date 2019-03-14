"""
Модуль функций сглаживания.
:Authors:
    - Vladimir Voronov
"""


from typing import Union, Tuple


Number = Union[int, float]


class Smoothing:
    def __init__(self):
        self._sigma = 0

    def avr(self, avr_type, *args, **kwargs):
        if avr_type in ('std', 'std_avr', 'standard'):
            return std_avr(*args, **kwargs)
        elif avr_type in ('efi', 'efi_avr'):
            new_a, self._sigma = efi_avr(*args, last_sigma=self._sigma, **kwargs)
            return new_a
        elif avr_type in ('moving', 'moving_avr'):
            return moving_avr(*args, **kwargs)
        else:
            raise ValueError('Переданное значение аргумента avr_type не поддерживается')


def std_avr(a: Number, last_a: Number, n: int, *args, **kwargs) -> Number:
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


def efi_avr(a: Number, last_a: Number, l: Number, *args, last_sigma: Number=0, **kwargs) -> Tuple[Number, Number]:
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


def moving_avr(a: Number, last_a: Number, last_k_a: Number, window_size: int, *args, **kwargs) -> Number:
    """
    Сглаживание методом скользящего среднего.
    :param a          : текущее значение коэффициента
    :type a           : int or float
    :param last_a     : предыдущее значение коэффициента
    :type last_a      : int or float
    :param last_k_a   : значение коэффициента на k-й итерации назад
    :type last_k_a    : int or float
    :param window_size: количество усредняемых значений
    :type window_size : int
    :return: значение коэффициента после сглаживания
    :rtype : int or float
    """
    return last_a + (1 / window_size) * (a - last_k_a)


_alias_map = {
    'standard': ('std', 'std_avr'),
    'efi': ('efi_avr',),
    'moving': ('moving_avr',),
}
