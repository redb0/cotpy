import copy

import sympy as sp

from settings import variable_names


class Regulator:
    def __init__(self, model):
        self._model = model

        self._expr = self._model.model_expr

        self._forecasts = []
        self._forecasts_vars = []  # [(x, u); ...]
        self._base_vars = []

        self._predicted_x = copy.deepcopy(self._model.outputs)
        self._predicted_u = copy.deepcopy(self._model.inputs)

        self._regulator_expr = None
        self._regulator_func = None

    def forecast_one_step(self):
        # скопировать группы x и u из модели
        print(self._expr)
        self.update_model(self._predicted_x)
        self.update_model(self._predicted_u)
        print(self._expr)
        self._forecasts.append(self._expr)
        self._forecasts_vars.append((copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u)))
        print('->', self._forecasts)
        print('->', self._forecasts_vars)
        # сделать прогноз на 1 такт вперед:
        # продвинуть тао у переменных на 1,
        # сменить имя новой переменной и заменить ее в sympy выражении модели
        # обновить численное значение переменной

    def update_model(self, variables):
        for g in variables:
            for obj in g.variables:
                old_name = obj.name
                obj.update_name_tao(obj.tao - 1)
                self.var_replace(old_name, obj.name)

    def copy_vars(self, dst, src):
        for i in range(len(dst)):
            for obj in src[i].variables:
                dst[i].add_var(copy.copy(obj))

    def expr_subs(self):
        for i in range(len(self._predicted_x)):
            for j in range(len(self._predicted_x[i].variables)):
                obj = self._predicted_x[i].variables[j]
                print('NAME ->', obj.get_tex_name())
                if obj.tao == 1 and obj.name[:len(variable_names['predicted_outputs'])] == variable_names['predicted_outputs']:
                    print(self._forecasts[0])
                    self._expr = self._expr.subs({sp.var(obj.name): self._forecasts[0]})
                    obj.copy_var(self._base_vars[0][i].variables[j])
                    self.copy_vars(self._predicted_u, self._base_vars[1])
            self._predicted_x[i].sorted_var()
        print(self._expr)

    def var_replace(self, old_name, new_name):
        self._expr = self._expr.subs({sp.var(old_name): sp.var(new_name)})

    def synthesis(self):
        # синтез закона управления
        if not self._predicted_u:
            pass
        max_u_tao = max([g.max_tao for g in self._predicted_u])
        min_tao: int = min([g.min_tao for g in self._predicted_u])
        print('u max tao', max_u_tao)
        print('u min tao', min_tao)
        step = 0
        while step < min_tao:
            print('-' * 30)
            self.forecast_one_step()
            self.expr_subs()
            if step == 0:
                self._base_vars = (copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u))
            step += 1

    def get_u(self):
        # рассчитать управляющее воздействие
        pass

    @property
    def expr(self):
        return self._regulator_expr

    @property
    def func(self):
        return self._regulator_func

    @property
    def model(self):
        return self._model


def main():
    import model
    m = model.create_model('a_0+a1*x1(t-1)+a_2*u1(t-3)')
    a = [[1, 2, 3], [5, 6, 7], [9, 0, 1]]
    x = [[1, 2, 3]]
    u = [[1, 2, 3]]
    # m = model.create_model('a_0*x1(t-1)+a_2*x1(t-3)+a_1*x1(t-2)+a3*u1(t-1)')
    # a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]
    # x = [[1, 2, 3, 4]]
    # u = [[1, 2, 3, 4]]
    m.initialization(a, x, u, type_memory='min')

    print('-'*20)
    r = Regulator(m)
    # r.forecast_one_step()
    print('-' * 30)
    # r.forecast_one_step()
    r.synthesis()
    print(len(r._predicted_u[0].variables))
    print(len(r._predicted_x[0].variables))

if __name__ == '__main__':
    main()
