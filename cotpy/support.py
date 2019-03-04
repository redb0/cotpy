"""
Модуль вспомогательных функций.
:Authors:
    - Vladimir Voronov
"""

import warnings

import collections


def flatten(l):
    """
    Функция для расплющивания многомерных списков.
    
    >>> list(flatten([1, [2, [3, 4]]]))
    [1, 2, 3, 4]
    
    >>> list(flatten([]))
    []
    
    >>> list(flatten([[1, 2], [3, 4]]))
    [1, 2, 3, 4]
    
    >>> list(flatten([[1, 'abc'], [3.5, 'd']]))
    [1, 'abc', 3.5, 'd']
    
    :param l: итерируемый объект различной вложенности
    :type l : iterable
    :return : генератор, последовательно выдающий элементы из многомерного списка.
    :rtype  : generator
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def is_rect_matrix(m, sub_len=0, min_len=0) -> bool:
    l = len(m[0])
    for val in m:
        if min_len:
            if len(val) < min_len:
                return False
        else:
            if len(val) != (sub_len if sub_len else l):
                return False
    return True


def normalize_kwargs(kw, alias_map=None, strict_compliance=False):
    """
    Вспомогательная функция для нормализации именованных аргументов.
    
    >>> normalize_kwargs({'a': 5, 'b': 10, 'c': 15}, alias_map={'abc': ['c', 'b', 'a']})
    {'abc': 15}
    
    Также будет возвращено предупреждение:
    UserWarning: Все псевдонимы kwargs ['c', 'b', 'a'] относятся к 'abc'. Будет применен только 'c'
    
    >>> normalize_kwargs({'a': 5, 'b': 10, 'c': 15}, alias_map={})
    {'a': 5, 'b': 10, 'c': 15}
    
    >>> normalize_kwargs({'a': 5, 'b': 10}, alias_map={'c': ['a'], 'd': ['b']})
    {'c': 5, 'd': 10}
    
    >>> normalize_kwargs({}, alias_map={'c': ['a'], 'd': ['b']})
    {}
    
    >>> normalize_kwargs({}, alias_map={})
    {}
    
    :param kw       : Словарь исходных именованных аргументов.
    :type kw        : dict
    :param alias_map: Отображение между каноническим именем и списком 
                      сокращений в порядке приоритета от большего к меньшему.
                      Если каноническое значение отсутствует в списке, 
                      предполагается, что оно имеет наивысший приоритет.
    :type alias_map : dict, optional
    :param strict_compliance: Флаг строго соответсвия ключей из kw и alias_map. 
                              Если имеются ключи, которых нет в alias_map или 
                              которые указаны в alias_map, поднимается исключение.
    :type strict_compliance: bool
    :return: Словарь нормализованных именованных аргументов.
    :rtype : dict
    """
    # по убыванию приоритета
    res = dict()
    if alias_map is None:
        alias_map = dict()
    for canonical, alias in alias_map.items():
        values, seen_key = [], []
        if canonical not in alias:
            if canonical in kw:
                values.append(kw.pop(canonical))
                seen_key.append(canonical)
        for a in alias:
            if a in kw:
                values.append(kw.pop(a))
                seen_key.append(a)
        if values:
            res[canonical] = values[0]
            if len(values) > 1:
                warnings.warn(f'Все псевдонимы kwargs {seen_key!r} относятся к '
                              f'{canonical!r}. Будет применен только {seen_key[0]!r}')
    res.update(kw)

    fail_keys = [k for k in res.keys() if k not in alias_map]
    if strict_compliance and fail_keys:
        raise TypeError(f"Не разрешенные ключи {fail_keys} в kwargs")
    fail_keys = [k for k in alias_map if k not in res.keys()]
    if strict_compliance and fail_keys:
        raise TypeError(f"Ключей {fail_keys} нет в kwargs")

    return res
