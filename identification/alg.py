from abc import abstractmethod
import numpy as np

import support
from identification.cores import cores_dict

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
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass


class Adaptive(Algorithm):   # 6.6
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

    def update(self, outputs_val, inputs_val, **kwargs):  # outputs_val(obj_val), inputs_val
        # TODO: возможно убрать inputs_val
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        m = self._identifier.model
        last_a = np.array(m.last_a)
        print('last_a', last_a)
        print('obj_val', outputs_val)
        grad = np.array(m.get_grad_value(*support.flatten(m.get_x_values()),
                                         *support.flatten(m.get_u_values()),
                                         *last_a))  # FIXME: проверить
        print('grad', grad)
        if self._method in ['simplest', 'smp']:
            new_a = Adaptive.simplest(last_a, *outputs_val, grad, **kw)  # gamma=g, g_type=g_type,
            # self._identifier.update_x(outputs_val)
            # self._identifier.update_a(new_a)
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
            print('MODEL', model_val)
            discrepancy = outputs_val - model_val
            new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad, self._matrix_k, weight, _lambda)
            # self._identifier.update_x(outputs_val)
            # self._identifier.update_a(new_a)
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
            grad_ah = np.array(m.get_grad_value(*support.flatten(m.get_x_values()),
                                                *support.flatten(m.get_u_values()),
                                                *self._last_ah))
            model_ah = m.func_model(*support.flatten(m.get_x_values()),
                                    *support.flatten(m.get_u_values()),
                                    *self._last_ah)
            print('model_ah', model_ah)
            n = self._identifier.n
            discrepancy = outputs_val - model_ah
            new_a, self._last_ah = Adaptive.pole(last_a, self._last_ah, grad_ah, discrepancy, n, gamma, weight)
            # self._identifier.update_x(outputs_val)
            # self._identifier.update_a(new_a)
            return new_a
        else:
            raise ValueError('Метода "' + self._method + '" не существует.')

    def init_k_matrix(self, n, init_weight):
        ar_grad = np.zeros((n, len(self._identifier.model.grad)))
        for i in range(n):
            x = self._identifier.model.get_outputs_value(i)
            u = self._identifier.model.get_inputs_value(i)
            a = self._identifier.model.get_coefficients_value(i)
            ar_grad[i] = np.array(self._identifier.model.get_grad_value(*x, *u, *a))
        self._matrix_k = Adaptive.find_initial_k(ar_grad, init_weight)

    @staticmethod
    def simplest(last_a, obj_val, grad, gamma=1, g_type='factor'):
        # простейший адаптивный алгоритм
        fraction = (obj_val - grad @ last_a)
        if gamma > 0:
            if g_type in ['factor', 'f']:
                fraction *= gamma
                fraction /= (grad @ grad)  # скаляр
            elif g_type in ['addend', 'a']:
                fraction /= (gamma + (grad @ grad))
            else:
                raise ValueError('Не найдено типа g_type (' + g_type + ').')
        else:
            raise ValueError('Gamma должна быть > 0. ' + str(gamma) + ' <= 0')
        new_a = last_a + fraction * grad  # вектор
        return new_a

    @classmethod
    def lsm(cls, last_a, discrepancy, grad, k_matrix, weight, _lambda=1):  # obj_val, model_val
        gamma = cls.find_gamma(grad, k_matrix, weight, _lambda)
        new_k = cls.find_k_matrix(grad, gamma, k_matrix, _lambda)
        new_a = last_a + gamma * discrepancy  # (obj_val - model_val)  # grad @ last_a
        return new_a, new_k

    @staticmethod
    def pole(last_a, last_ah, grad_ah, discrepancy, n, gamma, weight=1):  # obj_val, model_ah
        _gamma = gamma * np.power(n, -1/2)
        print('gamma =', _gamma)
        new_ah = last_ah + _gamma * (1 / weight) * grad_ah * discrepancy  # (obj_val - model_ah)
        print('new_ah =', new_ah)
        new_a = last_a + (1 / n) * (new_ah - last_a)
        return new_a, new_ah

    @staticmethod
    def find_gamma(grad, k_matrix, weight, _lambda=1):
        return np.dot(k_matrix, grad) / (weight * _lambda + np.dot(grad.T, np.dot(k_matrix, grad)))

    @staticmethod
    def find_k_matrix(grad, gamma, last_k_matrix, _lambda=1):
        e = np.eye(len(gamma))
        return ((e - np.dot(gamma[None, :].T, grad[None, :])) @ last_k_matrix) * (1 / _lambda)

    @staticmethod
    def find_initial_k(ar_grad, weight):
        # 6.6.4
        # weight должен быть в виде (s^2)
        # веса как число и как массив
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
    def __init__(self, identifier, **kwargs):
        super().__init__(identifier, **kwargs)

    def update(self, outputs_val, inputs_val, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        mu = 1 if 'mu' not in kw else kw['mu']
        if ('cores' in kw) and (kw['cores'] in cores_dict.keys()):
            cores_key = kw['cores']
        else:
            cores_key = list(cores_dict.keys())[0]
        m = self._identifier.model
        last_a = np.array(m.last_a)
        grad = np.array(m.get_grad_value(*support.flatten(m.get_x_values()),
                                         *support.flatten(m.get_u_values()),
                                         *last_a))
        print('last_a', last_a)
        print('obj_val', outputs_val)
        print('grad', grad)

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
            print('MODEL', model_val)
            discrepancy = outputs_val - model_val
            new_a = None
            if self._method == 'lsm':  # 6.7.1
                e = outputs_val - model_val
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=False)
                new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad,
                                                     self._matrix_k, kernel_func(e), _lambda)
            elif self._method == 'lsm_cipra':  # 6.7.7
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=True)
                new_a, self._matrix_k = AdaptiveRobust.lsm_cipra(last_a, discrepancy, grad,
                                                                 self._matrix_k, kernel_func, weight, _lambda)
            elif self._method == 'lsm_diff':  # 6.7.4
                kernel_func = cores_dict[cores_key](weight, mu, is_diff=True)
                discrepancy = kernel_func(discrepancy)
                new_a, self._matrix_k = Adaptive.lsm(last_a, discrepancy, grad, self._matrix_k, weight, _lambda)

            # self._identifier.update_x(outputs_val)
            # self._identifier.update_a(new_a)
            return new_a

        elif self._method == 'pole':  # 6.7.6
            gamma = 1 if 'gamma' not in kw else kw['gamma']
            kernel_func = cores_dict[cores_key](1, mu, is_diff=True)
            if self._last_ah is None:
                self._last_ah = np.array([1 for _ in range(len(last_a))])

            grad_ah = np.array(m.get_grad_value(*support.flatten(m.get_x_values()),
                                                *support.flatten(m.get_u_values()),
                                                *self._last_ah))
            model_ah = m.func_model(*support.flatten(m.get_x_values()),
                                    *support.flatten(m.get_u_values()), *self._last_ah)
            print('model_ah', model_ah)
            n = self._identifier.n
            discrepancy = kernel_func(outputs_val - model_ah)
            new_a, self._last_ah = Adaptive.pole(last_a, self._last_ah, grad_ah, discrepancy, n, gamma, weight=1)
            # self._identifier.update_x(outputs_val)
            # self._identifier.update_a(new_a)
            return new_a

    @classmethod
    def lsm_cipra(cls, last_a, discrepancy, grad, g_matrix, kernel_func, weight, _lambda=1):  # 6.7.7
        sigma = 1 / (weight * _lambda + np.dot(grad.T, np.dot(g_matrix, grad)))
        gamma = np.dot(g_matrix, grad) * sigma
        new_g = cls.find_k_matrix(grad, gamma, g_matrix, _lambda)
        new_a = last_a + gamma * kernel_func(sigma * discrepancy)
        return new_a, new_g

