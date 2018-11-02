from abc import abstractmethod
import numpy as np

import support

_alias_map = {
    'gamma': ['g'],
    'gamma_type': ['gt']
}


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
    # if orthogonality_matrix is None:
    #     orthogonality_matrix = np.eye(len(k))
    # k = k * orthogonality_matrix

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
    def __init__(self, identifier, method='smp'):
        super().__init__()
        self._identifier = identifier
        self._method = method

    def update(self, outputs_val, inputs_val, **kwargs):  # outputs_val(obj_val), inputs_val
        if self._method in ['simplest', 'smp']:
            # last_a, obj_val, grad, gamma = 1, g_type = 'factor'
            kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
            # if 'gamma' in kw:
            #     g = kw['gamma']
            # if 'gamma_type' in kw:
            #     g_type = kw['gamma_type']

            last_a = np.array(self._identifier.model.last_a)
            print('last_a', last_a)
            print('obj_val', outputs_val)
            grad = np.array(self._identifier.model.get_grad_value(*self._identifier.model.last_x, *inputs_val, *last_a))
            print('grad', grad)
            new_a = Adaptive.simplest(last_a, *outputs_val, grad, **kw)  # gamma=g, g_type=g_type,
            self._identifier.update_data(a=new_a, x=outputs_val, u=inputs_val)
            return new_a
        elif self._method == 'lsm':
            pass
        elif self._method == 'pole':
            pass
        else:
            raise ValueError('Метода "' + self._method + '" не существует.')

    @staticmethod
    def simplest(last_a, obj_val, grad, gamma=1, g_type='factor'):
        # простейший адаптивный алгоритм
        # TODO: документацию
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

    def cov_matrix(self):
        pass

    @classmethod
    def lsm(cls, last_a, obj_val, grad, k_matrix, weight):
        gamma = cls.find_gamma(grad, k_matrix, weight)
        new_k = cls.find_k_matrix(grad, gamma, k_matrix)
        new_a = last_a + gamma @ (obj_val - grad @ last_a)
        return new_a, new_k

    def pole(self):
        pass

    @staticmethod
    def find_gamma(grad, k_matrix, weight):
        return np.dot(k_matrix, grad) / (weight + np.dot(grad.T, np.dot(k_matrix, grad)))

    @staticmethod
    def find_k_matrix(grad, gamma, last_k_matrix):
        e = np.eye(len(gamma))
        return (e - np.dot(gamma, grad.T)) @ last_k_matrix

    @staticmethod
    def find_initial_k(ar_grad, weight):  # TODO: доразобрать ??????
        # 6.6.4
        # веса как число и как массив
        # ar_grad = [grad1, grad2, ...], grad1 = [grad_a1, grad_a2, ...]
        weight_ar = None
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
            pass

        k0 = weight_ar @ scalar_product(ar_grad.T, ar_grad, 1)



class AdaptiveRobust(Adaptive):  # 6.7
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass
