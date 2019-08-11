import numpy as np
from cotpy import support

_alias_map = {
    'method': ['m'],
    'k0': ['k', 'k_matrix'],
    'init_n': ['n0', 'n'],
    'weight': ['w', 'p', 'sigma'],
    'init_weight': ['iw'],
    'efi_lambda': ['efi', 'l'],
    'init_ah': ['ah'],
    'gamma': ['g'],
    'g_type': ['gt', 'gamma_type'],

    'use_increments': ['use_inc', 'uinc'],
    'use_memory': ['um'],

    'wfm': ['weight_fm'],
    'core': ['nf'],

    'deep_tuning': ['deept', 'dt'],
    'deviation': ['h'],
    # 'adaptive_weight': ['aw']
}


def adjust_fm(a, discrepancy, init_w, nf=None):  # is_adaptive_w=False
    if nf:
        w = nf(discrepancy - a)
    else:
        w = init_w
    return a + w * (discrepancy - a)


class IRobust:
    def get_weight(self, *args, **kwargs):
        pass

    def get_nf(self, *args, **kwargs):
        pass


class Robust(IRobust):
    def get_weight(self, weight, e, s=None, delta=None, wt=0):
        if wt == 0:
            if s:
                return weight * abs(e) ** s  # if s >= 0
            raise AttributeError('Не указано значение атрибута s')
        elif wt == 1:
            if not delta:
                raise AttributeError('Не указано значение атрибута delta')
            if abs(e) <= delta:
                return weight
            else:
                return weight * abs(e) * (1 / delta)
        else:
            raise AttributeError('Некорректное значение атрибута wt')

    def get_nf(self, e, s=None, delta=None, nft=0):
        if nft == 0:
            return np.sign(e)
        elif nft == 1:
            if s:
                return np.sign(e) * abs(e) ** s
            raise AttributeError('Не указано значение атрибута s')
        elif nft == 2:
            if not delta:
                raise AttributeError('Не указано значение атрибута delta')
            if abs(e) <= delta:
                return e
            else:
                return delta * np.sign(e)
        else:
            raise AttributeError('Некорректное значение атрибута nft')


class NotRobust(IRobust):
    def get_weight(self, weight):
        return weight

    def get_nf(self, e):
        return e


# class AdaptiveWeight:  # Добавлять как класс или как отдельную функцию
#     @staticmethod
#     def adjust_fm(a, discrepancy, init_w, nf=None):  # is_adaptive_w=False
#         if nf:
#             w = nf(discrepancy - a)
#         else:
#             w = init_w
#         return a + w * (discrepancy - a)


class Algorithm:
    def __init__(self):
        self._ir = None

    def update(self, *args, **kwargs):
        pass

    def set_robust_behavior(self, ir: IRobust) -> None:
        self._ir = ir


class Recurrent(Algorithm):
    def __init__(self):
        super().__init__()


class Retrospective(Algorithm):
    def __init__(self):
        super().__init__()


class SMP(Recurrent):
    def __init__(self, identifier, **kwargs):
        super().__init__()
        self._identifier = identifier
        self.set_robust_behavior(NotRobust())

    def update(self, outputs_val, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        m = self._identifier.model
        last_a = np.array(m.last_a)
        grad = np.array(m.get_grad_value())

        deep_tuning = False if 'deep_tuning' not in kw else kw['deep_tuning']
        use_memory = False if 'use_memory' not in kw else kw['use_memory']

        wfm = 0.95 if 'wfm' not in kw else kw['wfm']
        core = None if 'core' not in kw else kw['core']

        if use_memory:
            use_increments = False if 'use_increments' not in kw else kw['use_increments']
            memory = self._identifier.model.memory_size
            if memory <= 0:
                raise ValueError('Величина памяти должна быть > 0')
            dh = np.array(self._identifier.model.outputs_values(memory))
            fi = np.zeros((memory, len(last_a)))
            for i in range(memory):
                fi[i] = np.array(m.get_grad_value(coefficient=last_a, **m.get_var_values(n=-i)))
            dh[1:] = dh[:-1]
            dh[0] = outputs_val[0]

            if use_increments:
                fi1 = np.zeros((memory - 1, len(last_a)))
                dh1 = np.zeros(len(dh) - 1)
                for i in range(memory - 1):
                    fi1[i] = fi[i] - fi[i + 1]

                for i in range(len(dh) - 1):
                    dh1[i] = dh[i] - dh[i + 1]
                new_a = self.__class__.smp_m(last_a, dh, fi)

                idx_fm = m.get_index_fm()
                if idx_fm is not None:
                    a0 = dh[0] - m.get_value(new_a) + grad[idx_fm] * new_a[idx_fm]  # a0 = dh[0] - m.get_value(new_a)
                    # a0 = outputs_val[0] - m.get_value(new_a) + grad[idx_fm] * new_a[idx_fm]
                    new_a[idx_fm] = adjust_fm(new_a[idx_fm], a0, wfm, core)
            else:
                new_a = self.__class__.smp_m(last_a, dh, fi)
        else:
            h = 0.1 if 'deviation' not in kw else kw['deviation']
            new_a = last_a.copy()
            if deep_tuning:
                i = 0
                delta = 2

                while abs(delta) >= 2 * h or i == 0:
                    grad = np.array(m.get_grad_value(coefficient=new_a))
                    grad_p = np.array(m.get_grad_value(coefficient=new_a, **m.get_var_values(n=-1)))
                    d_grad = grad - grad_p
                    if all(map(lambda y: y <= np.power(10.0, -10), d_grad)):  # np.finfo(float).eps
                        break
                    delta = outputs_val[0] - m.last_x[0] - d_grad @ new_a
                    # delta -= h * np.sign(delta)
                    new_a = self.__class__.smp(new_a, delta, d_grad, gamma=1, g_type='factor')  # **kw

                    idx_fm = m.get_index_fm()
                    if idx_fm is not None:
                        a0 = outputs_val[0] - m.get_value(new_a) + grad[idx_fm] * new_a[idx_fm]
                        new_a[idx_fm] = adjust_fm(new_a[idx_fm], a0, wfm, core)
                    i += 1
                    # print('число глубоких подстроек =', i)

            discrepancy = outputs_val - m.get_value(new_a)
            # discrepancy -= h * np.sign(discrepancy)
            new_a = self.__class__.smp(new_a, *discrepancy, grad, **kw)  # gamma=g, g_type=g_type,

        return new_a

    @staticmethod
    def smp_m(last_a, outputs, grads_matrix):
        # delta = np.linalg.pinv(grads_matrix) @ (outputs - grads_matrix.T @ last_a)
        # delta = (outputs - last_a @ grads_matrix.T) @ np.linalg.pinv(grads_matrix)
        delta = np.linalg.pinv(grads_matrix.T @ grads_matrix) @ grads_matrix.T @ (outputs - last_a @ grads_matrix.T)
        return last_a + delta

    @staticmethod
    def smp(last_a, discrepancy, grad, gamma=1, g_type='factor', **kwargs):
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


class RecurLSM(Recurrent):
    def __init__(self, identifier, **kwargs):
        super().__init__()
        self._identifier = identifier
        self._k = None
        self.set_robust_behavior(NotRobust())

    def update(self, outputs_val, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        m = self._identifier.model
        last_a = np.array(m.last_a)
        grad = np.array(m.get_grad_value())

        weight = 1 if 'weight' not in kw else kw['weight']
        init_weight = 1 if 'init_weight' not in kw else kw['init_weight']
        _lambda = 1 if 'efi_lambda' not in kw else kw['efi_lambda']
        use_increments = False if 'use_increments' not in kw else kw['use_increments']

        wfm = 0.95 if 'wfm' not in kw else kw['wfm']
        core = None if 'core' not in kw else kw['core']

        if self._k is None:
            if self._identifier.n0 != 0:
                self.find_init_k(self._identifier.n0, init_weight)
            else:
                # TODO: добавить автоматическое заполнение, или определение матрици как h*E
                raise ValueError('Не проведена инициализация начальных значений.')

        if use_increments:
            grad_p = m.get_grad_value(coefficient=last_a, **m.get_var_values(n=-1))
            d_grad = grad - grad_p
            delta = outputs_val[0] - m.last_x[0] - d_grad @ last_a  # d_grad[1:] @ last_a[1:]
            new_a, self._k = self.__class__.lsm(last_a, delta, d_grad, self._k, weight, _lambda)  # **kw

            idx_fm = m.get_index_fm()
            if idx_fm is not None:
                a0 = outputs_val[0] - m.get_value(new_a) + grad[idx_fm] * new_a[idx_fm]
                new_a[idx_fm] = adjust_fm(new_a[idx_fm], a0, wfm, core)  # wfm - weight_fm, nf - core
        else:
            discrepancy = outputs_val - m.get_value(last_a)
            new_a, self._k = self.__class__.lsm(last_a, discrepancy, grad, self._k, weight, _lambda)
        return new_a

    # @staticmethod
    # def adjust_fm(a, discrepancy, init_w, nf=None):  # is_adaptive_w=False
    #     if nf:
    #         w = nf(discrepancy - a)
    #     else:
    #         w = init_w
    #     return a + w * (discrepancy - a)

    @classmethod
    def lsm(cls, last_a, discrepancy, grad, k_matrix, weight, _lambda=1):
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
        gamma = np.dot(k_matrix, grad) / cls.find_sigma(grad, k_matrix, weight, _lambda)
        new_k = cls.find_k(grad, gamma, k_matrix, _lambda)
        new_a = last_a + gamma * discrepancy  # (obj_val - model_val)  # grad @ last_a
        return new_a, new_k

    @staticmethod
    def find_sigma(grad, k_matrix, weight, _lambda=1):
        """
        Формула в формате TeX:
        \gamma_{n} = \frac{K_{a_{n-1}}\nabla_{a}h}{p_{n}\lambda + \nabla_{a}^{T}hK_{a_{n-1}}\nabla_{a}h},
        h = h(u_{n}, a_{n-1})
        """
        return weight * _lambda + np.dot(grad.T, np.dot(k_matrix, grad))

    @staticmethod
    def find_k(grad, gamma, last_k_matrix, _lambda=1):
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

    def find_init_k(self, n: int, weight) -> None:
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
            ar_grad[i] = m.get_grad_value(coefficient=m.get_coefficients_value(i),
                                          **m.get_var_values(n=i + 1))

        if isinstance(weight, (int, float)):
            weight = np.full(n, weight)
        elif isinstance(weight, (list, np.ndarray)):
            if len(weight) != n:
                raise ValueError(f'Длина списка значений весов не совпадает с длиной списка градиетов: '
                                 f'{len(weight)} != {n}.')
            if isinstance(weight, list):
                weight = np.array(weight)
        else:
            raise TypeError('Аргумент weight некорректного типа.')

        self._k = self.__class__._find_init_k(ar_grad, weight)

    @classmethod
    def _find_init_k(cls, ar_grad, weight):
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
        return (1 / weight) * ar_grad.T @ ar_grad  # np.linalg.inv(k)


class RecurRobustLSM(RecurLSM):
    def __init__(self, identifier, **kwargs):
        super().__init__(identifier, **kwargs)
        self.set_robust_behavior(Robust())

    def update(self, outputs_val, **kwargs):
        pass


class RecurPole(Recurrent):
    pass


class RetrospLSM(Retrospective):
    def __init__(self, identifier, **kwargs):
        super().__init__()
        self._identifier = identifier
        self.set_robust_behavior(NotRobust())

    def update(self, inputs, outputs, *args, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=_alias_map)
        w = 0.01 if 'weight' not in kw else kw['weight']
        m = self._identifier.model
        new_a = None

        use_increments = False if 'use_increments' not in kw else kw['use_increments']

        if isinstance(w, (list, np.ndarray)):
            if len(w) != len(outputs):
                raise ValueError('Длина вектора весов не совпадает с коллиеством измерений')
        if isinstance(w, list):
            w = np.array(w)

        if use_increments:  # sequential linearization method
            # TODO: Проверить на правильность
            number_iter = 5 if 'numit' not in kw else kw['numit']
            # gamma = 1
            new_a = np.array(m.last_a)
            init_data = m.get_all_var_values()
            deviation = 0
            for i in range(number_iter):
                ar_grad = np.zeros((len(outputs), len(m.coefficients)))
                h = np.zeros((len(outputs)))
                for j in range(len(ar_grad)):
                    ar_grad[j] = m.get_grad_value()  # коэф, выходы, входы
                    h[j] = outputs[j] - m.get_last_model_value()
                    m.update_x([outputs[j]])
                    m.update_u([inputs[j]])
                delta = self.__class__.lsm(ar_grad, h, w)
                new_a = new_a + (1 / (i + 1)) * delta  # (gamma / (i + 1))
                d = np.sum(np.power(h, 2))  # / weight
                # print(d)
                if i != 0 and deviation < d:
                    new_a = m.last_a
                    break
                deviation = d
                m.update_a(new_a)
                m.initialization(type_memory='min', memory_size=0, **init_data)
        else:  # least square method
            ar_grad = np.zeros((len(outputs), len(m.coefficients)))
            for i in range(len(ar_grad)):
                ar_grad[i] = m.get_grad_value()  # коэф, выходы, входы
                m.update_x([outputs[i]])
                m.update_u([inputs[i]])
            new_a = self.__class__.lsm(ar_grad, outputs, w)
        return new_a

    @classmethod
    def lsm(cls, ar_grad: np.ndarray, ar_outputs_val: np.ndarray, w):
        # некоррелированная равноточная
        matrix_f = (ar_grad.T * (1 / w)) @ ar_grad  # scalar_product(ar_grad.T, ar_grad, w)
        matrix_h = (ar_grad.T * (1 / w)) @ ar_outputs_val  # scalar_product(ar_grad.T, ar_outputs_val, w)
        new_a = np.linalg.solve(matrix_f, matrix_h)
        if np.allclose(np.dot(matrix_f, new_a), matrix_h):
            return new_a
        else:
            raise np.linalg.LinAlgError()

    # def _scalar_product(self, x, y, w):
    #     if isinstance(w, (list, np.ndarray)):
    #         if len(w) != len(x):
    #             raise ValueError(f'Количество измерений != длине вектора весов: {len(x)} != {len(w)}.')
    #     if isinstance(w, list):
    #         w = np.array(w)
    #
    #     return (x * (1 / w)) @ y


class RobustLSM(RetrospLSM):
    pass
