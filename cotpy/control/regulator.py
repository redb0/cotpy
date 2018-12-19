"""
Модуль классов синтеза регулятора на основе модели объекта.
:Authors:
    - Vladimir Voronov
"""

import copy

import numpy as np
import sympy as sp
from cotpy.settings import variable_names
from sympy.utilities.autowrap import ufuncify

from cotpy import support, model


class Regulator:
    """
    Класс синтеза закона адаптивного управления.
    """
    def __init__(self, m):
        """
        Атрибуты:
        _expr              : sympy выражение модели объекта
        _desired_output_sp : sympy переменная желаемой траектории из variable_names['trajectory']
        _forecasts         : список sympy выражений прогнозов на каждом шаге синтеза
        _forecasts_vars    : список кортежей переменных (выходы, входы) на каждом шаге синтеза
        _base_vars         : переменные (выходы, входы), состоящие в выражении при прогнозе на первый такт вперед
        _predicted_x       : текущие переменные выходов исплозующиеся во время синтеза закона управления
        _predicted_u       : текущие переменные входов исплозующиеся во время синтеза закона управления
        _regulator_expr    : sympy выражение закона управления
        _regulator_func    : функция расчета управляющего воздействия (numpy.ufunc)
        _args              : список аргументов функции расчета управляющео воздействия в виде sympy переменных
        _low_limit         : ограничение снизу на управление
        _high_limit        : ограничение сверху на управление
        
        :param m: экземпляр класса Model, содержащий модель объекта
        :type m : class Model from model.py
        """
        self._model = m
        self._expr = self._model.sp_expr
        self._desired_output_sp = sp.var(variable_names['trajectory'])
        self._forecasts = []
        self._forecasts_vars = []  # [(x, u); ...]
        self._base_vars = []
        self._predicted_x = copy.deepcopy(self._model.outputs)
        self._predicted_u = copy.deepcopy(self._model.inputs)
        self._regulator_expr = None
        self._regulator_func = None
        self._args = []
        self._low_limit = None
        self._high_limit = None

    def apply_restrictions(self, u):
        """
        Метод накладывания ограничений на управление
        :param u: значение управляющего воздействия
        :type u : number or list
        :return: None
        """
        if self._low_limit is not None:
            for i in range(len(u)):
                if isinstance(self._low_limit, (list, np.ndarray)):
                    limit = self._low_limit[i]
                else:
                    limit = self._low_limit
                if u[i] <= limit:
                    u[i] = limit

        if self._high_limit is not None:
            for i in range(len(u)):
                if isinstance(self._low_limit, (list, np.ndarray)):
                    limit = self._high_limit[i]
                else:
                    limit = self._high_limit
                if u[i] >= limit:
                    u[i] = limit

    def update(self, output, setpoint, *args, **kwargs):
        """
        Метод расчета управляющего воздействия.
        :param output        : значение выхода объекта
        :type output         : number or list
        :param setpoint: значение уставки (желаемой трактории движения объекта)
        :type setpoint : number or list
        :param args          : -
        :param kwargs        : -
        :return: значение управляющего воздействия
        :rtype : number or list
        :raises TypeError: если output не является числом или списком
        """
        x_val = []
        xs = self._model.outputs
        for i in range(len(self._predicted_x)):
            v = self._predicted_x[i].variables
            for j in range(len(v)):
                if v[j].tao != 0:
                    x_val.append(xs[i].values[-v[j].tao])
        u_val = []
        us = self._model.inputs
        for i in range(len(self._predicted_u)):
            v = self._predicted_u[i].variables
            for j in range(len(v)):
                if v[j].tao != 0:
                    u_val.append(us[i].values[-v[j].tao])
        last_a = np.array(self._model.last_a)

        if isinstance(setpoint, (int, float)):
            setpoint = [setpoint]
        if isinstance(output, (int, float)):
            output = [output]
        if isinstance(output, (list, np.ndarray)):
            if len(output) == 1 and len(self._predicted_x) == 1:
                u = self._regulator_func(*setpoint, *output, *x_val, *u_val, *last_a)
                u = np.array([u])
                self.apply_restrictions(u)
                return u
            elif len(output) > 1 and len(self._predicted_x) > 1:
                u = self._regulator_func(*setpoint, *output, *x_val, *u_val, *last_a)
                u = np.array([u])
                self.apply_restrictions(u)
                return u
            else:
                TypeError(f'Аргумент output некорректного типа: {type(output)}.')

    def forecast_one_step(self):
        """
        Метод делает прогноз на 1 такт вперед.
        Обновляет переменные входа (u), выхода (x), с
        охраняет полученное выражение в self._forecasts, 
        а переменные в self._forecasts_vars.
        :return: None
        """
        self.update_expr(self._predicted_x)
        self.update_expr(self._predicted_u)
        self._forecasts.append(self._expr)
        self._forecasts_vars.append((copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u)))

    def update_expr(self, variables):
        """
        Метод делает обновляет выражение модели, 
        делая проноз по группе переменных variables.
        Происходи обновление запаздывания переменной tao = tao + 1, и соответственно имени.
        Например:
        Variable('x0_2', 2) -> Variable('x0_1', 1)
        :param variables: список экземпляров класса GroupVariable.
        :return: None
        """
        for g in variables:
            for obj in g.variables:
                old_name = obj.name
                obj.update_name_tao(obj.tao - 1)
                self.var_replace(old_name, obj.name)

    def copy_vars(self, dst, src):
        """
        Копирует переменные из списка групп src в список групп dst.
        :param dst: список групп переменных куда будут копироваться переменные из списка src.
        :type dst: список экземпляров класса GroupVariable.
        :param src: список источник.
        :type src: список экземпляров класса GroupVariable.
        :return: None
        """
        for i in range(len(dst)):
            for obj in src[i].variables:
                if obj.tao not in list([j.tao for j in dst[i].variables]):
                    dst[i].add_var(copy.copy(obj))

    def expr_subs(self):
        """
        Метод производит замену прогнозной выходной переменной 
        на выражение полученное на предыдущем шаге, т.е. если
        при синтезе в выражении появилась переменная x(t+1):
        
        модель: y(t|a(t)) = f(x(t-1), u(t-1-tao), a(t))
        первая итерация: y(t+1|a(t)) = f(x(t), u(t-tao), a(t))
        вторая итерация: y(t+2|a(t)) = f(x(t+1), u(t-tao+1), a(t))
        
        переменная x(t+1) это тоже самое что y(t+1|a(t)), т.е. x(t+1)=y(t+1|a(t)), 
        идет замена: 
        y(t+2|a(t)) = f(y(t+1|a(t)), u(t-tao), a(t)) -> 
        -> y(t+2|a(t)) = f(f(x(t+1), u(t-tao), a(t)), u(t-tao+1), a(t))
        
        :return:  None
        """
        for i in range(len(self._predicted_x)):
            for j in range(len(self._predicted_x[i].variables)):
                obj = self._predicted_x[i].variables[j]
                if obj.tao == 1 and obj.name[:len(variable_names['predicted_outputs'])] == variable_names['predicted_outputs']:
                    self._expr = self._expr.subs({sp.var(obj.name): self._forecasts[0]})
                    obj.copy_var(self._base_vars[0][i].variables[j])
                    self.copy_vars(self._predicted_u, self._base_vars[1])
            self._predicted_x[i].sorted_var()

    def var_replace(self, old_name, new_name):
        """Метод замены переменных в sympy выражении модели"""
        self._expr = self._expr.subs({sp.var(old_name): sp.var(new_name)})

    def synthesis(self) -> None:
        """
        Метод синтеза закона управления.
        Закон управления минтезируется путем составления последовательных прогнозов 
        на некоторое количество тактов вперед, пока в выражении не появиться u(t).
        После чего этот прогноз приравнивается к желаемой траектории на данном 
        такте и из получившегося равенства выражается управление(u(t)).
        
        Если объект описывается уравнением: 
        x(t) = f(x(t-1), u(t-1-tao), a) + e(t), t=1,2,3,...
        то модель будет выглядеть следующим образом:
        y(t|a(t)) = f(x(t-1), u(t-1-tao), a(t))
        Прогноз на 1 такт вперед:
        y(t+1|a(t)) = f(x(t), u(t-tao), a(t))
        повторение процедуры пока не появиться u(t):
        y(t+1+tao|a(t)) = f(y(t+tao|a(t)), u(t), a(t))
        Приравнивание у желаемой трактории:
        y(t+1+tao|a(t)) = x(t+1+tao)*
        
        Выражение закона управления сохраняется в self._regulator_expr,
        а функция в self._regulator_func.

        :return: None
        """
        if not self._predicted_u:
            pass
        min_tao: int = min([g.min_tao for g in self._predicted_u])
        step = 0
        while step < min_tao:
            self.forecast_one_step()
            self.expr_subs()
            if step == 0:
                self._base_vars = (copy.deepcopy(self._predicted_x), copy.deepcopy(self._predicted_u))
            step += 1

        sp_vars_x_current = sp.var([i[0].name for i in (g.variables for g in self._predicted_x) if i[0].tao == 0])
        sp_vars_x = sp.var([i.name for i in support.flatten([g.variables for g in self._predicted_x]) if i.tao != 0])
        sp_vars_u = sp.var([i.name for i in support.flatten([g.variables for g in self._predicted_u])])
        sp_vars_a = sp.var([i.name for i in self._model.coefficients])

        self._args = [self._desired_output_sp, *sp_vars_x_current, *sp_vars_x, *sp_vars_u[1:], *sp_vars_a]
        expr = sp.solve(self._expr - self._desired_output_sp, sp_vars_u[0])
        self._regulator_expr = expr[0]
        # print('Выражение:', expr[0])
        # print('Аргументы:', self._args)
        self._regulator_func = ufuncify(self._args, expr[0])

    def set_limit(self, low, high) -> None:
        """
        Установка ограничений на управление.
        :param low : Значение нижней границы управления
        :type low  : number or list
        :param high: Значение верхней границы управления
        :type high : number or list
        :return: None
        """
        self._high_limit = high
        self._low_limit = low

    @property
    def expr(self):
        """Sympy выражение закона управления"""
        return self._regulator_expr

    @property
    def func(self):
        """Функция расчета управляющего воздействия"""
        return self._regulator_func

    @property
    def model(self):
        """"Экземпляр класса Model с моделью объекта"""
        return self._model

    @property
    def forecasts(self):
        return self._forecasts

    @property
    def forecasts_vars(self):
        return self._forecasts_vars

    @property
    def low_limit(self):
        return self._low_limit

    @low_limit.setter
    def low_limit(self, v) -> None:
        self._low_limit = v

    @property
    def high_limit(self):
        return self._high_limit

    @high_limit.setter
    def high_limit(self, v) -> None:
        self._high_limit = v
