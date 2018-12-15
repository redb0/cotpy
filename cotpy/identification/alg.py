"""
Модуль классов с алгоритмами идентификации.
Содержит:
    - Адаптивные алгоритмы идентификации
    - Робастные адаптивные алгоритмы

:Authors:
    - Vladimir Voronov
"""

from abc import abstractmethod

import numpy as np
from cotpy import support

from cotpy.identification.cores import cores_dict

_alias_map = {
    'method': ['m'],
    'k0': ['k', 'k_matrix'],
    'init_n': ['n0', 'n'],
    'weight': ['w', 'p', 'sigma'],
    'init_weight': ['iw'],
    'efi_lambda': ['efi', 'l'],
    'init_ah': ['ah'],
    'gamma': ['g'],
    'gamma_type': ['gt']
}


# def scalar_product(m1: np.ndarray, m2: np.ndarray, weight: Union[Number, ListNumber, np.ndarray]) -> np.ndarray: ...
def scalar_product(m1, m2, weight):  # orthogonality_matrix=None
    # 6.3.6, 6.3.8
    n_0 = len(m1)
    if isinstance(weight, (int, float)):
        weight = np.array([1 / weight for _ in m1])
        # k = (1 / weight) * (m1 @ m2)  # ar_grad.T @ ar_grad
    elif isinstance(weight, (list, np.ndarray)):
        if n_0 != len(weight):
            raise ValueError(f'Количество измерений не совпадает: {n_0} != {len(weight)}.')
        # k = (m1 * (1 / weight)) @ m2  # (ar_grad.T * weight) @ ar_grad
    else:
        raise TypeError('Неверный тип аргумента weight. необходим int, float, ndarray.')

    k = (m1 * (1 / weight)) @ m2  # (ar_grad.T * weight) @ ar_grad

    return k


class Algorithm:
    def __init__(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class LSM(Algorithm):  # п 6.3, 6.4
    """Класс алгоритмов МНК."""
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        # TODO: написать обертку для lsm
        pass

    @staticmethod
    def lsm(ar_grad, ar_outputs_val, weight):
        # некоррелированная равноточная
        matrix_f = scalar_product(ar_grad.T, ar_grad, weight)
        matrix_h = scalar_product(ar_grad.T, ar_outputs_val, weight)
        new_a = np.linalg.solve(matrix_f, matrix_h)
        if np.allclose(np.dot(matrix_f, new_a), matrix_h):
            return new_a
        else:
            raise np.linalg.LinAlgError()


class Robust(Algorithm):  # 6.5
    """Класс робастных алгоритмов идентификации"""
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass


class Adaptive(Algorithm):   # 6.6
    """Класс адаптивных алгоритмов идентификации."""
    def __init__(self, identifier, **kwargs):
        super().__init__()
        self._identifier = identifier
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        if 'method' in kw:
            self._method = kw['method']
        else:
            raise AttributeError('Не указан метод идентификации.')
        self._matrix_k = None if 'k0' not in kw else kw['k0']  # на старте k0 далее k
        self._n0 = None if 'init_n' not in kw else kw['init_n']  # начальное кол-во измерений >= кол-ву alpha

        self._last_ah = None if 'init_ah' not in kw else kw['init_ah']

    def update(self, outputs_val, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        m = self._identifier.model
        last_a = np.array(m.last_a)
        grad = np.array(m.get_grad_value(*list(support.flatten(m.get_x_values())),
                                         *list(support.flatten(m.get_u_values())),
                                         *last_a))  # FIXME: проверить
        if self._method in ['simplest', 'smp']:
            model_val = m.get_last_model_value()
            discrepancy = outputs_val - model_val
            new_a = Adaptive.simplest(last_a, *discrepancy, grad, **kw)  # gamma=g, g_type=g_type,
            return new_a
        elif self._method == 'lsm':  # 6.6.3, 6.6.8
            weight = 1 if 'weight' not in kw else kw['weight']
            init_weight = 1 if 'init_weight' not in kw else kw['init_weight']
            _lambda = 1 if 'efi_lambda' not in kw else kw['efi_lambda']

            if self._matrix_k is None:
                if self._identifier.n0 != 0:
                    self.init_k_matrix(self._identifier.n0, init_weight)
                else:
                    # TODO: добавить автоматическое заполнение
                    raise ValueError('Не проведена инициализация начальных значений.')

            model_val = m.get_last_model_value()  # FIXME: проверить
            discrepancy = outputs_val - model_val
            new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad, self._matrix_k, weight, _lambda)
            return new_a

        elif self._method == 'pole':
            weight = 1
            gamma = 1
            if 'weight' in kw:
                weight = kw['weight']
            if 'gamma' in kw:
                gamma = kw['gamma']
            if self._last_ah is None:
                self._last_ah = np.array([1 for _ in range(len(last_a))])
            grad_ah = np.array(m.get_grad_value(*list(support.flatten(m.get_x_values())),
                                                *list(support.flatten(m.get_u_values())),
                                                *self._last_ah))
            model_ah = m.func_model(*list(support.flatten(m.get_x_values())),
                                    *list(support.flatten(m.get_u_values())),
                                    *self._last_ah)
            n = self._identifier.n
            discrepancy = outputs_val - model_ah
            new_a, self._last_ah = Adaptive.pole(last_a, self._last_ah, grad_ah, discrepancy, n, gamma, weight)
            return new_a
        else:
            raise ValueError('Метода "' + self._method + '" не существует.')

    def init_k_matrix(self, n: int, init_weight) -> None:
        """
        Обертка для Adaptive.find_initial_k.
        
        :param n          : число начальных измерений
        :type n           : int
        :param init_weight: веса.
        :type init_weight : number, list, np.ndarray
        :return: None
        """
        m = self._identifier.model
        ar_grad = np.zeros((n, len(m.grad)))
        for i in range(n):
            ar_grad[i] = np.array(m.get_grad_value(*list(support.flatten(m.get_x_values(n=i))),
                                                   *list(support.flatten(m.get_u_values(n=i))),
                                                   *np.array(m.get_coefficients_value(i))))
        self._matrix_k = Adaptive.find_initial_k(ar_grad, init_weight)

    @staticmethod
    def simplest(last_a, discrepancy, grad, gamma=1, g_type='factor'):
        """
        Простейший адаптивный алгоритм.
        
        Математическое описание в формате TeX:
        a_{n} = a_{n-1} + \frac{(h_{n}^{*}-g^T(u_{n})a_{n-1})}{(g^T(u_{n})g(u_{n})}g(u_{n})
        
        где: n          - номер итерации;
             a_{n}      - новое (искомое) значение коэыыициентов;
             a_{n-1}    - предыдущее значение коэффициентов;
             h_{n}^{*}  - значение выхода объекта на итерации n;
             u_{n}      - управляющее воздействие;
             g(u_{n})   - вектор базисных функций;
             g^T(u_{n}) - транспонированный вектор базисных функций.
             
        :param last_a : список значений коэффикиентов на прошлой итерации.
        :type last_a  : list, np.ndarray
        :param discrepancy: невязка, равная разнице выходя объекта и модели, т.е. h_{n}^{*} - h(u_{n}, a_{n-1}).
        :type discrepancy : number
        :param grad   : Список значений базисных функций.
        :type grad    : numpy.ndarray
        :param gamma  : Коэффициент для уменьшения чувствительности к помехам.
        :type gamma   : number
        :param g_type : Метод ввода коэффициента gamma.
                        Варианты:
                            1) ['factor', 'f'] - как множитель.
                            2) ['addend', 'a'] - как слогаемое в знаменатели дроби.
        :type g_type  : str
        :return: Список новых значений коэффициентов.
        :rtype : numpy.ndarray
        """
        # TODO: сделать для нелинейных моделей.
        # fraction = (obj_val - grad @ last_a)
        if gamma > 0:
            if g_type in ['factor', 'f']:
                discrepancy *= gamma
                discrepancy /= (grad @ grad)  # скаляр
            elif g_type in ['addend', 'a']:
                discrepancy /= (gamma + (grad @ grad))
            else:
                raise ValueError('Не найдено типа g_type (' + g_type + ').')
        else:
            raise ValueError('Gamma должна быть > 0. ' + str(gamma) + ' <= 0')
        new_a = last_a + discrepancy * grad  # вектор
        return new_a

    @classmethod
    def lsm(cls, last_a, discrepancy, grad, k_matrix, weight, _lambda=1):  # obj_val, model_val
        """
        Адаптивный метод наименьших квадратов.
        
        Математическое описание в формате TeX:
        a_{n} = a_{n-1} + \gamma_{n}[h_{n}^{*} - h(u_{n}, a_{n-1})],
        h = h(u_{n}, a_{n-1}),
        \gamma_{n} = \frac{K_{a_{n-1}}\nabla_{a}h}{p_{n}\lambda + \nabla_{a}^{T}hK_{a_{n-1}}\nabla_{a}h}
        K_{a_{n}} = [E - \gamma_{n}\nabla_{a}^{T}h]K_{a_{n-1}}\lambda^{-1}, n = n_{0}+1, n_{0}+2, ...
        
        K_{a_{n_{0}}} = \sum_{i=1}^{n_{0}}{p_{i}^{-1}\nabla_{a}h(u_{i}, a_{i})\nabla_{a}^{T}h(u_{i}, a_{i})}, 
        i = 1, ..., n_{0}
        
        где: \nabla_{a}h - вектор градиента по параметрам a;
             \lambda     - коэффициент забывания информации 0<\lambda<=1;
             E           - едининичная матрица;
             p           - вес, при p = \sigma^{2} алгоритм становиться стандартный адаптивным МНК.
             n_{0}       - начальное колличество измерений, n_{0} >= кол-во коэффициентов a.
        
        :param last_a     : список значений коэффикиентов на прошлой итерации.
        :type last_a      : list, np.ndarray
        :param discrepancy: невязка, равная разнице выходя объекта и модели, т.е. [h_{n}^{*} - h(u_{n}, a_{n-1})].
        :type discrepancy : number
        :param grad       : список значений градиента по параметрам a.
        :type grad        : np.ndarray
        :param k_matrix   : матрица K на предудущей интерции.
        :type k_matrix    : np.ndarray
        :param weight     : значение веса.
        :type weight      : number
        :param _lambda    : значение коэффициента забывания информации.
        :type _lambda     : number
        :return: кортеж из списка с новыми коэфиициетами и матрицей K.
        :rtype : Tuple(np.ndarray, np.ndarray)
        """
        gamma = cls.find_gamma(grad, k_matrix, weight, _lambda)
        new_k = cls.find_k_matrix(grad, gamma, k_matrix, _lambda)
        new_a = last_a + gamma * discrepancy  # (obj_val - model_val)  # grad @ last_a
        return new_a, new_k

    @staticmethod
    def pole(last_a, last_ah, grad_ah, discrepancy, n, gamma, weight=1):  # obj_val, model_ah
        # FIXME: работает некорректно
        _gamma = gamma * np.power(n, -1/2)
        # print('gamma =', _gamma)
        new_ah = last_ah + _gamma * (1 / weight) * grad_ah * discrepancy  # (obj_val - model_ah)
        # print('new_ah =', new_ah)
        new_a = last_a + (1 / n) * (new_ah - last_a)
        return new_a, new_ah

    @staticmethod
    def find_gamma(grad, k_matrix, weight, _lambda=1):
        """
        Расчет параметра gamma для алгоритмов квадратичного критерия.
        
        Формула в формате TeX:
        \gamma_{n} = \frac{K_{a_{n-1}}\nabla_{a}h}{p_{n}\lambda + \nabla_{a}^{T}hK_{a_{n-1}}\nabla_{a}h},
        h = h(u_{n}, a_{n-1})
        
        :param grad    : список значений градиента по параметрам a.
        :type grad     : np.ndarray
        :param k_matrix: матрица K.
        :type k_matrix : np.ndarray
        :param weight  : вес.
        :type weight   : number
        :param _lambda : коэффициент забывания информации.
        :type _lambda  : number
        :return: список значений gamma.
        :rtype : np.ndarray
        """
        return np.dot(k_matrix, grad) / (weight * _lambda + np.dot(grad.T, np.dot(k_matrix, grad)))

    @staticmethod
    def find_k_matrix(grad, gamma, last_k_matrix, _lambda=1):
        """
        Расчет матрицы K.
        
        Формула в формате TeX:
        K_{a_{n}} = [E - \gamma_{n}\nabla_{a}^{T}h]K_{a_{n-1}}\lambda^{-1}, 
        n = n_{0}+1, n_{0}+2, ...
        
        :param grad         : список значений градиента по параметрам a.
        :type grad          : np.ndarray
        :param gamma        : список значений gamma.
        :type gamma         : np.ndarray
        :param last_k_matrix: матрица K на предудущей интерции.
        :type last_k_matrix : np.ndarray
        :param _lambda      : коэффициент забывания информации.
        :type _lambda       : number
        :return: матрица K.
        :rtype : np.ndarray
        """
        e = np.eye(len(gamma))
        return ((e - np.dot(gamma[None, :].T, grad[None, :])) @ last_k_matrix) * (1 / _lambda)

    @staticmethod
    def find_initial_k(ar_grad, weight): # 6.6.4
        """
        Расчет начальной матрицы K.
        
        Формула в формате TeX:
        K_{a_{n_{0}}} = \sum_{i=1}^{n_{0}}{p_{i}^{-1}\nabla_{a}h(u_{i}, a_{i})\nabla_{a}^{T}h(u_{i}, a_{i})}, 
        i = 1, ..., n_{0}
        
        :param ar_grad: массив значений градиента для каждого измерения i.
        :type ar_grad : np.ndarray
        :param weight : вес.
        :type weight  : number, list, np.ndarray
        :return: начальное значение матрицы K.
        :rtype : np.ndarray
        :raises ValueError: если длина списка weight не совпадает с длиной ar_grad.
        :raises TypeError : если weight не является одним из типов number, list, np.ndarray.
        """
        # ar_grad = [grad1, grad2, ...], grad1 = [grad_a1, grad_a2, ...]
        n0 = len(ar_grad)
        if isinstance(weight, (int, float)):
            weight_ar = np.array([weight for _ in range(n0)])
        elif isinstance(weight, np.ndarray):
            weight_ar = weight
            if len(weight) != n0:
                raise ValueError(f'Длина списка значений весов не совпадает с длиной списка градиетов: '
                                 f'{len(weight)} != {n0}.')
        elif isinstance(weight, list):
            weight_ar = np.array(weight)
            if len(weight) != n0:
                raise ValueError(f'Длина списка значений весов не совпадает с длиной списка градиетов: '
                                 f'{len(weight)} != {n0}.')
        else:
            raise TypeError('Аргумент weight некорректного типа.')

        k = (1 / weight_ar) * ar_grad.T @ ar_grad
        return k  # np.linalg.inv(k)


class AdaptiveRobust(Adaptive):  # 6.7
    """Класс адаптивных робастных методов идентификации."""
    def __init__(self, identifier, **kwargs):
        super().__init__(identifier, **kwargs)

    def update(self, outputs_val, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        mu = 1 if 'mu' not in kw else kw['mu']
        if ('core' in kw) and (kw['core'] in cores_dict.keys()):
            cores_key = kw['core']
        else:
            cores_key = list(cores_dict.keys())[0]
        m = self._identifier.model
        last_a = np.array(m.last_a)
        grad = np.array(m.get_grad_value(*list(support.flatten(m.get_x_values())),
                                         *list(support.flatten(m.get_u_values())),
                                         *last_a))

        if self._method in ('lsm', 'lsm_cipra', 'lsm_diff'):
            _lambda = 1 if 'efi_lambda' not in kw else kw['efi_lambda']
            weight = 1 if 'weight' not in kw else kw['weight']
            init_weight = 1 if 'init_weight' not in kw else kw['init_weight']
            if self._matrix_k is None:
                n0 = self._identifier.n0
                if n0 != 0:
                    self.init_k_matrix(self._identifier.n0, init_weight)
                else:
                    raise ValueError('Не проведена инициализация начальных значений.')

            model_val = m.get_last_model_value()
            discrepancy = outputs_val - model_val
            new_a = None
            if self._method == 'lsm':  # 6.7.1
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=False)
                new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad,
                                                     self._matrix_k, kernel_func(discrepancy), _lambda)
            elif self._method == 'lsm_cipra':  # 6.7.7
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=True)
                new_a, self._matrix_k = AdaptiveRobust.lsm_cipra(last_a, discrepancy, grad,
                                                                 self._matrix_k, kernel_func, weight, _lambda)
            elif self._method == 'lsm_diff':  # 6.7.4
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=True)
                discrepancy = kernel_func(discrepancy)
                new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad, self._matrix_k, weight, _lambda)
            return new_a

        elif self._method == 'pole':  # 6.7.6
            gamma = 1 if 'gamma' not in kw else kw['gamma']
            kernel_func = cores_dict[cores_key](1, mu, is_diff=True)
            if self._last_ah is None:
                self._last_ah = np.array([1 for _ in range(len(last_a))])

            grad_ah = np.array(m.get_grad_value(*list(support.flatten(m.get_x_values())),
                                                *list(support.flatten(m.get_u_values())),
                                                *self._last_ah))
            model_ah = m.func_model(*list(support.flatten(m.get_x_values())),
                                    *list(support.flatten(m.get_u_values())), *self._last_ah)
            n = self._identifier.n
            discrepancy = kernel_func(outputs_val - model_ah)
            new_a, self._last_ah = Adaptive.pole(last_a, self._last_ah, grad_ah, discrepancy, n, gamma, weight=1)
            return new_a

    @classmethod
    def lsm_cipra(cls, last_a, discrepancy, grad, g_matrix, kernel_func, weight, _lambda=1):  # 6.7.7
        sigma = 1 / (weight * _lambda + np.dot(grad.T, np.dot(g_matrix, grad)))
        gamma = np.dot(g_matrix, grad) * sigma
        new_g = cls.find_k_matrix(grad, gamma, g_matrix, _lambda)
        new_a = last_a + gamma * kernel_func(sigma * discrepancy)
        return new_a, new_g

