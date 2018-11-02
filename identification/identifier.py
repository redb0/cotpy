class Identifier:
    def __init__(self, model) -> None:
        self._model = model

        # self._sigma = sigma  # сразу в алгоритм даже стартовые значения
        self._n0 = 0
        self._n = 0  # используется при обычном сглаживании
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

    def init_data(self, a=None, x=None, u=None, type_memory='min'):
        self._model.initialization(a, x, u, type_memory=type_memory)
        if a:
            self._n0 = len(a[0])
        else:
            self._n0 = len(self._model.coefficients)

    def __repr__(self):
        return f'Identifier({repr(self._model)}, n0={self._n0})'

    def __str__(self):
        pass

    @property
    def model(self):
        return self._model

    @property
    def n(self):
        return self._n

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


def main():
    # TODO: добавить проверку на максимальное tao.
    import model

    model = model.create_model('a_0*x(t-1)+a_2*u(t-1)+a_1*x(t-2)')
    # model = model.create_model('a_0*x1(t-1)+a_3*x2(t-3)+a_2*x2(t-1)+a_1*x1(t-2)')
    # model = model.create_model('a_2*u(t-1)+a1+a3')
    print(model.model_expr_str)
    print(model.model_expr)
    print(model.inputs)
    print(model.outputs)
    print(model.coefficients)
    print('-' * 20)
    iden = Identifier(model)
    a = [[1, 2, 3], [5, 6, 7], [9, 0, 1]]
    x = [[1, 2, 3]]
    u = [[5, 6, 7]]
    iden.init_data(a, x, u, type_memory='min')
    # iden.init_data()
    print(iden.model.a_values)
    print(iden.model.x_values)
    print(iden.model.u_values)
    print('-' * 20)
    iden.update_data(a=[10, 20, 30], x=[50], u=[40])
    print('a', iden.model.a_values)
    print('x', iden.model.x_values)
    print('u', iden.model.u_values)
    print('-' * 20)
    print(iden.model.last_a)
    print(iden.model.last_x)
    print(iden.model.last_u)
    print('-' * 20)


if __name__ == '__main__':
    main()
