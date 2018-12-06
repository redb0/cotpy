"""
Модуль классов с алгоритмами идентификации.
Содержит:
    - Адаптивные алгоритмы идентификации
    - Робастные адаптивные алгоритмы

:Authors:
    - Vladimir Voronov
"""


class Identifier:
    """Класс идентификатора."""
    def __init__(self, model) -> None:
        self._model = model

        # self._sigma = sigma  # сразу в алгоритм даже стартовые значения
        self._n0 = 0
        self._n = 1  # используется при обычном сглаживании
        self._last_cov = None
        self._delta = 0  # используется при сглаживании методом экспоненциального забывания информации
        self._last_ap = None

    def min_numb_measurements(self):
        len_a = len(self._model.coefficients)
        nm_u = [(i.group_name, len_a + i.max_tao) for i in self._model.inputs]
        nm_x = [(i.group_name, len_a + i.max_tao) for i in self._model.outputs]
        return ('a', len_a), nm_x, nm_u

    def update_data(self, a, u, x):
        self._model.update_data(a, u, x)
        self._n += 1

    def update_a(self, a):
        self._model.update_a(a)
        self._n += 1

    def update_u(self, u):
        self._model.update_u(u)

    def update_x(self, x):
        self._model.update_x(x)

    def init_data(self, a=None, x=None, u=None, type_memory='max'):
        self._model.initialization(a, x, u, type_memory=type_memory)
        if a:
            self._n0 = len(a[0])
        else:
            self._n0 = len(self._model.coefficients)

    def __repr__(self):
        return f'Identifier({repr(self._model)}, n0={self._n0})'

    @property
    def model(self):
        return self._model

    @property
    def n(self) -> int:
        return self._n

    @property
    def n0(self) -> int:
        return self._n0

    @property
    def last_cov(self):
        return self._last_cov

    @last_cov.setter
    def last_cov(self, val):
        self._last_cov = val

    @property
    def last_ap(self):
        return self._last_ap

    @last_ap.setter
    def last_ap(self, val):
        self._last_ap = val

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, val):
        self._delta = val