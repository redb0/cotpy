import support


class Identifier:
    def __init__(self, model, sigma=0):
        self._model = model

        self._memory_len = 0
        self._sigma = sigma

    def init_zeros(self, n0=0):
        len_a = len(self._model.coefficients)
        len_u = len(self._model.inputs)
        len_x = len(self._model.outputs)
        self._memory_len = n0 if n0 else len_a
        a = [[0. for _ in range(self._memory_len)] for _ in range(len_a)]
        u = [[0. for _ in range(self._memory_len)] for _ in range(len_u)]
        x = [[0. for _ in range(self._memory_len)] for _ in range(len_x)]
        self.initialization(a, x, u, n0=self._memory_len)

    def initialization(self, a, x, u, n0=0) -> None:  # TODO: добавить типы
        sub_len = n0 if n0 and n0 >= len(a[0]) else len(a[0])
        self.init_a(a, sub_len)
        self.init_u(u, sub_len)
        self.init_x(x, sub_len)

    def init_a(self, a, n0=0) -> None:  # a - двумерный массив
        # TODO: возможно добавить заполнения из dict - 'name': [v1, v2, ...]
        l = len(a)
        if l == len(self._model.coefficients):
            self._memory_len = n0
            if support.is_rect_matrix(a, sub_len=n0):
                for i in range(l):
                    self._model.coefficients[i].initialization(a[i])
            else:
                raise ValueError('Подмассивы должны быть одинаковой длины')
        else:
            raise ValueError('Несоответствие количества коэффициентов и длины массива a. ' +
                             str(l) + ' != ' + str(len(self._model.coefficients)))

    def init_u(self, u, n0=0) -> None:  # TODO: добавить типы
        l = len(u)
        if l == len(self._model.inputs):
            if support.is_rect_matrix(u, sub_len=n0):
                for i in range(l):
                    for var in self._model.inputs[i]:
                        var.initialization(u[i])
            else:
                raise ValueError('Подмассивы должны быть одинаковой длины')
        else:
            raise ValueError('Несоответствие количества входов и длины массива u. ' +
                             str(l) + ' != ' + str(len(self._model.inputs)))

    def init_x(self, x, n0=0) -> None:  # TODO: добавить типы
        if len(x) == len(self._model.outputs):
            if support.is_rect_matrix(x, sub_len=n0):
                for i in range(len(x)):
                    for var in self._model.outputs[i]:
                        var.initialization(x[i])
            else:
                raise ValueError('Подмассивы должны быть одинаковой длины')
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
    print(model.model_expr_str)
    print(model.model_expr)
    print(model.inputs)
    print(model.outputs)
    print(model.coefficients)

    iden = Identifier(model)
    iden.init_zeros()
    print(iden.a_values)
    print(iden.x_values)
    print(iden.u_values)
    iden.init_x([[1, 2, 3]])
    print(iden.x_values)

    iden.model.outputs[0][0].values[0] = 10
    print(iden.x_values)


if __name__ == '__main__':
    main()
