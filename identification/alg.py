from abc import abstractmethod
import numpy as np


class Algorithm:
    def __init__(self):
        pass

    @abstractmethod
    def find_a(self, identifier, method, *args, **kwargs):
        pass


class LSM(Algorithm):
    def __init__(self):
        super().__init__()

    def find_a(self, identifier, method, *args, **kwargs):
        pass


class Robust(Algorithm):
    def __init__(self):
        super().__init__()

    def find_a(self, identifier, method, *args, **kwargs):
        pass


class Adaptive(Algorithm):
    def __init__(self):
        super().__init__()

    @staticmethod
    def find_a(identifier, method, *args, **kwargs):
        if method == 'simplest' or method == 'smp':
            g_type = 'factor'
            g = 1
            obj_val = None
            u_n = None
            if args and len(args) >= 2:
                obj_val = args[0]
                u_n = args[1]
            elif 'obj_val' in kwargs:
                obj_val = kwargs['obj_val']
            else:
                raise AttributeError('Не передано значение объекта')
            if 'g' in kwargs:
                g = kwargs['g']
            elif 'gamma' in kwargs:
                g = kwargs['g']
            if 'gt' in kwargs:
                g_type = kwargs['gt']
            elif 'g_type' in kwargs:
                g_type = kwargs['g_type']

            last_a = np.array(identifier.model.last_a)
            print('last_a', last_a)
            print('obj_val', obj_val)
            grad = np.array(identifier.model.get_grad_value(*identifier.model.last_x, *u_n, *last_a))
            print('grad', grad)
            new_a = Adaptive.simplest(last_a, *obj_val, grad, gamma=g, g_type=g_type)
            return new_a
        elif method == 'lsm':
            pass
        elif method == 'pole':
            pass
        else:
            raise ValueError('Метода "' + method + '" не существует.')

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

    def find_a(self, identifier, method, *args, **kwargs):
        pass
