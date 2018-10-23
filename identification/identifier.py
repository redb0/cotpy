import support


class Identifier:
    def __init__(self, model, sigma=0.) -> None:
        self._model = model

        self._memory_len = 0
        self._sigma = sigma

    def min_numb_measurements(self):
        len_a = len(self._model.coefficients)
        nm_u = [(i.group_name, len_a + i.max_tao) for i in self._model.inputs]
        nm_x = [(i.group_name, len_a + i.max_tao) for i in self._model.outputs]
        return ('a', len_a), nm_x, nm_u

    def init_zeros(self, n0: int=0) -> None:
        len_a = len(self._model.coefficients)
        len_u = len(self._model.inputs)
        len_x = len(self._model.outputs)
        n0 = n0 if n0 >= len_a else len_a
        a = [[0. for _ in range(n0)] for _ in range(len_a)]
        u = [[0. for _ in range(n0 + self._model.inputs[i].max_tao)] for i in range(len_u)]
        x = [[0. for _ in range(n0 + self._model.outputs[i].max_tao)] for i in range(len_x)]
        self.initialization(a, x, u)

    def initialization(self, a, x, u) -> None:  # TODO: добавить типы
        self._memory_len = len(a[0])
        if a:
            self.init_a(a)
        if u:
            self.init_u(u)
        if x:
            self.init_x(x)

    def init_a(self, a) -> None:  # a - двумерный массив
        # TODO: возможно добавить заполнения из dict - 'name': [v1, v2, ...]
        l = len(a)
        if l == len(self._model.coefficients):
            if support.is_rect_matrix(a, sub_len=self._memory_len):
                for i in range(l):
                    self._model.coefficients[i].initialization(a[i])
            else:
                raise ValueError('Подмассивы должны быть одинаковой длины')
        else:
            raise ValueError('Несоответствие количества коэффициентов и длины массива a. ' +
                             str(l) + ' != ' + str(len(self._model.coefficients)))

    def init_u(self, u) -> None:  # TODO: добавить типы
        l = len(u)
        if l == len(self._model.inputs):
            if support.is_rect_matrix(u, min_len=self._memory_len):
                for i in range(l):
                    for var in self._model.inputs[i]:
                        var.initialization(u[i])
            else:
                raise ValueError('Подмассивы должны быть длиной не менее ' + str(self._memory_len))
        else:
            raise ValueError('Несоответствие количества входов и длины массива u. ' +
                             str(l) + ' != ' + str(len(self._model.inputs)))

    def init_x(self, x) -> None:  # TODO: добавить типы
        if len(x) == len(self._model.outputs):
            if support.is_rect_matrix(x, min_len=self._memory_len):
                for i in range(len(x)):
                    for var in self._model.outputs[i]:
                        var.initialization(x[i])
            else:
                raise ValueError('Подмассивы должны быть длиной не менее ' + str(self._memory_len))
        else:
            raise ValueError('Несоответствие количества выходов и длины массива x. ' +
                             str(len(x)) + ' != ' + str(len(self._model.outputs)))

    def update_a(self, a) -> None:
        c = self._model.coefficients
        len_c = len(c)
        if len(a) != len_c:
            raise ValueError(f'Длина массива {a} должна быть = {len_c}. {len_c} != {len(a)}')
        for i in range(len_c):
            c[i].update(a[i])

    def update_x(self, val) -> None:
        x = self._model.outputs  # группы
        len_x = len(x)
        if len_x != len(val):
            raise ValueError(f'Длина массива {val} должна быть = {len_x}. {len_x} != {len(val)}')
        for i in range(len_x):
            x[i][0].update(val[i])

    def update_u(self, val) -> None:
        u = self._model.inputs  # группы
        len_u = len(u)
        if len_u != len(val):
            raise ValueError(f'Длина массива {val} должна быть = {len_u}. {len_u} != {len(val)}')
        for i in range(len_u):
            u[i][0].update(val[i])

    def __repr__(self):
        pass

    def __str__(self):
        pass

    @property
    def a_values(self):
        return [a.values for a in self._model.coefficients]

    @property
    def x_values(self):
        return [x.values for x in support.flatten(self._model.outputs)]

    @property
    def u_values(self):
        return [u.values for u in support.flatten(self._model.inputs)]

    @property
    def model(self):
        return self._model

    @property
    def memory_len(self):
        return self._memory_len

    @property
    def sigma(self):
        return self._sigma


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
    iden.init_zeros()
    print('a', iden.a_values)
    print('x', iden.x_values)
    print('u', iden.u_values)
    print('-' * 20)
    iden.init_a([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    iden.init_x([[6, 5, 4]])
    iden.init_u([[1, 2, 6]])
    print('new_a', iden.a_values)
    print('new_x', iden.x_values)
    print('new_u', iden.u_values)
    print('-' * 20)
    print(iden.min_numb_measurements())
    print('-' * 20)
    iden.update_a([1, 2, 3])
    print('a', iden.a_values)
    iden.update_x([5])
    print('x1', iden.x_values)
    iden.update_x([9])
    print('x2', iden.x_values)
    iden.update_u([10])
    print('u', iden.u_values)
    print('-' * 20)
    print(iden.model.last_a)
    print(iden.model.last_x)
    print(iden.model.last_u)
    print('-' * 20)
    iden.init_x([[6, 5]])


if __name__ == '__main__':
    main()
