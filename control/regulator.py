import copy

import numpy as np
import sympy as sp
from sympy.utilities.autowrap import ufuncify

from settings import variable_names
import support


class Regulator:
    def __init__(self, model):
        self._model = model

        self._expr = self._model.model_expr
        self._desired_output_sp = sp.var(variable_names['trajectory'])

        self._forecasts = []
        self._forecasts_vars = []  # [(x, u); ...]
        self._base_vars = []

        self._predicted_x = copy.deepcopy(self._model.outputs)
        # self._current_x = []  # x1_0, x2_0
        self._predicted_u = copy.deepcopy(self._model.inputs)

        self._regulator_expr = None
        self._regulator_func = None

        self._args = []
        self._args_in_func = []

    def update(self, output, desired_output, *args, **kwargs):
        x_val = []
        xs = self._model.outputs
        for i in range(len(self._predicted_x)):
            v = self._predicted_x[i].variables
            for j in range(len(v)):
                if v[j].tao != 0:
                    x_val.append(xs[i].values[-v[j].tao])
        print('Значения x', x_val)
        u_val = []
        us = self._model.outputs
        for i in range(len(self._predicted_u)):
            v = self._predicted_u[i].variables
            for j in range(len(v)):
                if v[j].tao != 0:
                    u_val.append(us[i].values[-v[j].tao])
        print('Значения u', u_val)
        print('x(t)', output)
        print('xt', desired_output)
        last_a = np.array(self._model.last_a)
        print('a', last_a)

        if isinstance(output, (int, float)) and len(self._predicted_x) == 1:
            if isinstance(desired_output, (int, float)):
                desired_output = [desired_output]
            return self._regulator_func(*desired_output, output, *x_val, *u_val, *last_a)
        elif isinstance(output, (list, np.ndarray)) and len(self._predicted_x) > 1:
            return self._regulator_func(*desired_output, *output, *x_val, *u_val, *last_a)
        else:
            TypeError(f'Аргумент output некорректного типа: {type(output)}.')

    def forecast_one_step(self):
        # скопировать группы x и u из модели
        self.update_expr(self._predicted_x)
        self.update_expr(self._predicted_u)
        self._forecasts.append(self._expr)
        self._forecasts_vars.append((copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u)))
        # сделать прогноз на 1 такт вперед:
        # продвинуть тао у переменных на 1,
        # сменить имя новой переменной и заменить ее в sympy выражении модели
        # обновить численное значение переменной

    def update_expr(self, variables):
        for g in variables:
            for obj in g.variables:
                old_name = obj.name
                obj.update_name_tao(obj.tao - 1)
                self.var_replace(old_name, obj.name)

    def copy_vars(self, dst, src):
        for i in range(len(dst)):
            for obj in src[i].variables:
                if obj.tao not in list([j.tao for j in dst[i].variables]):
                    dst[i].add_var(copy.copy(obj))

    def expr_subs(self):
        for i in range(len(self._predicted_x)):
            for j in range(len(self._predicted_x[i].variables)):
                obj = self._predicted_x[i].variables[j]
                if obj.tao == 1 and obj.name[:len(variable_names['predicted_outputs'])] == variable_names['predicted_outputs']:
                    self._expr = self._expr.subs({sp.var(obj.name): self._forecasts[0]})
                    obj.copy_var(self._base_vars[0][i].variables[j])
                    self.copy_vars(self._predicted_u, self._base_vars[1])
            self._predicted_x[i].sorted_var()

    def var_replace(self, old_name, new_name):
        self._expr = self._expr.subs({sp.var(old_name): sp.var(new_name)})

    def synthesis(self):
        # синтез закона управления
        if not self._predicted_u:
            pass
        min_tao: int = min([g.min_tao for g in self._predicted_u])
        step = 0
        while step < min_tao:
            print('-' * 30)
            self.forecast_one_step()
            self.expr_subs()
            if step == 0:
                self._base_vars = (copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u))
            step += 1

        sp_vars_x_current = sp.var([i[0].name for i in (g.variables for g in self._predicted_x) if i[0].tao == 0])
        print('current:', sp_vars_x_current)

        sp_vars_x = sp.var([i.name for i in support.flatten([g.variables for g in self._predicted_x]) if i.tao != 0])
        print('x:', sp_vars_x)
        sp_vars_u = sp.var([i.name for i in support.flatten([g.variables for g in self._predicted_u])])
        sp_vars_a = sp.var([i.name for i in self._model.coefficients])

        self._args = [self._desired_output_sp, *sp_vars_x_current, *sp_vars_x, *sp_vars_u[1:], *sp_vars_a]  # *sp_vars_x_current
        print('Аргументы:', self._args)

        expr = sp.solve(self._expr - self._desired_output_sp, sp_vars_u[0])
        print('Выражение:', expr[0])
        self._regulator_func = ufuncify(self._args, expr[0])

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
    # m = model.create_model('a_0*x1(t-1)+a2*u1(t-1)')
    # a = [[1, 2], [5, 6]]
    # x = [[1, 2]]
    # u = [[1, 2]]
    # m = model.create_model('a_0*x1(t-1)+a_1*x2(t-3)+a_2*u1(t-2)')  # не работает
    # a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]
    # x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    # u = [[3, 4, 5, 6]]
    # m = model.create_model('a_0*x1(t-2)+a2*x1(t-2)+a_1*u1(t-3)')  # нужно добавить упрощение и объединение коэффициентов
    # a = [[1, 2, 2], [5, 6, 3], [5, 6, 3]]
    # x = [[1, 2, 3]]
    # u = [[3, 4, 3]]
    m.initialization(a, x, u, type_memory='min')

    print('-'*20)
    r = Regulator(m)
    print('-' * 30)
    r.synthesis()
    u = r.update(5, 10)
    print(u)


if __name__ == '__main__':
    main()
