"""
Модуль классов синтеза регулятора на основе модели объекта.
:Authors:
    - Vladimir Voronov
"""

import copy

import numpy as np
import sympy as sp
from cotpy.settings import control_law_vars
from sympy.utilities.autowrap import ufuncify

from cotpy import support


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
        self._desired_output_sp = sp.var(control_law_vars['trajectory'])
        self._forecasts = []
        self._forecasts_vars = []  # [(x, u); ...]
        self._base_vars = {}
        var = self.model.model_vars
        self._predicted_vars = copy.deepcopy({k: var[k] for k in var.keys() if k != 'coefficient'})
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

    def update(self, output, setpoint, ainputs=(), *args, **kwargs):
        """
        Метод расчета управляющего воздействия.
        :param output  : значение выхода объекта
        :type output   : number or list
        :param setpoint: значение уставки (желаемой трактории движения объекта)
        :type setpoint : number or list
        :param ainputs : значения дополнительных входов модели (по умолчанию обозначенных 'z')
        :type ainputs  : number or list
        :param args    : -
        :param kwargs  : -
        :return: значение управляющего воздействия
        :rtype : number or list
        :raises TypeError: если output не является числом или списком
        """
        var_values = []
        variables = self._model.model_vars
        for k, v in self._predicted_vars.items():
            values = []
            for idx_group in range(len(v)):
                var_group = v[idx_group].variables
                for i in range(len(var_group)):
                    if var_group[i].tao != 0:
                        values.append(variables[k][idx_group].values[-var_group[i].tao])
            var_values.append(values)

        last_a = np.array(self._model.last_a)

        if isinstance(setpoint, (int, float)):
            setpoint = [setpoint]
        if isinstance(output, (int, float)):
            output = [output]
        if isinstance(ainputs, (int, float)):
            ainputs = [ainputs]
        u = self._regulator_func(*np.hstack(var_values), *last_a, *setpoint, *output, *ainputs)
        u = np.array([u])
        self.apply_restrictions(u)
        return u

    def forecast_one_step(self):
        """
        Метод делает прогноз на 1 такт вперед.
        Обновляет переменные входа (u), выхода (x), с
        охраняет полученное выражение в self._forecasts, 
        а переменные в self._forecasts_vars.
        :return: None
        """
        for k, v in self._predicted_vars.items():
            self.update_expr(v)

        self._forecasts.append(self._expr)
        self._forecasts_vars.append((copy.deepcopy(v) for v in self._predicted_vars.values()))

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
                    dst[i].add_vars(copy.copy(obj))

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
        predicted_x = self._predicted_vars['output']
        for idx_group in range(len(predicted_x)):  # по группам
            j = 0
            replacement = False
            while j < len(predicted_x[idx_group].variables):  # по переменным
                obj = predicted_x[idx_group].variables[j]
                if obj.tao == -1 and obj.name[:len(control_law_vars['predicted_output'])] == control_law_vars['predicted_output']:
                    self._expr = self._expr.subs({sp.var(obj.name): self._forecasts[0]})
                    predicted_x[idx_group].variables.pop(j)
                    replacement = True
                j += 1
            if replacement:
                for k, v in self._predicted_vars.items():
                    for i in range(len(v)):
                        v[i].add_unique_var(self._base_vars[k][i].variables)

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
        # if not self._predicted_u:
        #     pass
        min_tao: int = min([g.min_tao for g in self._predicted_vars['input']])
        # print('min_tao =', min_tao)
        step = 0
        while step < min_tao:
            # e = self._expr
            self.forecast_one_step()
            self.expr_subs()
            if step == 0:
                self._base_vars = {k: copy.deepcopy(v) for k, v in self._predicted_vars.items()}
            step += 1

        sp_var = []
        sp_current_vars = dict()
        for k, v in self._predicted_vars.items():
            sp_current_vars[k] = []
            for item in support.flatten([g.variables for g in v]):
                if item.tao != 0:
                    sp_var.append(sp.var(item.name))
                else:
                    sp_current_vars[k].append(sp.var(item.name))

        sp_vars_a = sp.var([i.name for i in self._model.coefficients])

        self._args = [*sp_var, *sp_vars_a, self._desired_output_sp,
                      *np.hstack([v for k, v in sp_current_vars.items() if k != 'input'])]
        expr = sp.solve(self._expr - self._desired_output_sp, sp_current_vars['input'][0])
        self._regulator_expr = expr[0]
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
    def expr_args(self):
        return self._args

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
