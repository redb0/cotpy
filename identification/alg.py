from abc import abstractmethod
import numpy as np

import support

_alias_map = {
    'gamma': ['g'],
    'gamma_type': ['gt']
}


class Algorithm:
    def __init__(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class LSM(Algorithm):
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass


class Robust(Algorithm):
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass


class Adaptive(Algorithm):
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

    def lsm(self):
        pass

    def pole(self):
        pass


class AdaptiveRobust(Adaptive):
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs):
        pass
