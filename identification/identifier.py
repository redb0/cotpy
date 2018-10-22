import support


class Identifier:
    def __init__(self, model, sigma=0):
        self._model = model

        self._memory_len = 0
        self._sigma = sigma

    def min_numb_measurements(self):
        len_a = len(self._model.coefficients)
        nm_u = [(i[0].name.split('_')[0], len_a + i[-1].tao) for i in self._model.inputs]
        nm_x = [(i[0].name.split('_')[0], len_a + i[-1].tao) for i in self._model.outputs]
        return ('a', len_a), nm_x, nm_u

    def init_zeros(self, n0=0):
        len_a = len(self._model.coefficients)
        len_u = len(self._model.inputs)
        len_x = len(self._model.outputs)
        n0 = n0 if n0 >= len_a else len_a
        # if not n0:
        #     self._memory_len = len_a
        # elif n0 and n0 >= len_a:
        #     self._memory_len = n0
        # else:
        #     raise ValueError('Минимальное количество значений коэффициентов должно быть >= ' + str(len_a))
        # a = [[0. for _ in range(self._memory_len)] for _ in range(len_a)]
        # u = [[0. for _ in range(self._memory_len + self._model.inputs[i][-1].tao)] for i in range(len_u)]
        # x = [[0. for _ in range(self._memory_len + self._model.outputs[i][-1].tao)] for i in range(len_x)]
        # self.initialization(a, x, u, n0=self._memory_len)
        a = [[0. for _ in range(n0)] for _ in range(len_a)]
        u = [[0. for _ in range(n0 + self._model.inputs[i][-1].tao)] for i in range(len_u)]
        x = [[0. for _ in range(n0 + self._model.outputs[i][-1].tao)] for i in range(len_x)]
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
    iden.init_u([[1, 2, 6]])
    print('new_a', iden.a_values)
    print('x', iden.x_values)
    print('new_u', iden.u_values)
    print('-' * 20)
    print(iden.min_numb_measurements())


if __name__ == '__main__':
    main()
