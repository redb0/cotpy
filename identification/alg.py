from abc import abstractmethod


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

    def find_a(self, identifier, method, *args, **kwargs):
        if method == 'simplest' or method == 'smp':
            pass
        elif method == 'lsm':
            pass
        elif method == 'pole':
            pass

    @staticmethod
    def simplest(last_a, obj_val, grad):
        # простейший адаптивный алгоритм
        # TODO: документацию
        fraction = (obj_val - grad @ last_a) / (grad @ grad)  # скаляр
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
