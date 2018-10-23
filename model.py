from typing import Union, List, Optional, NoReturn
import operator
import itertools

from sympy.utilities.autowrap import ufuncify
from sympy.core.add import Add
import sympy as sp

from analyzer.validation import check_brackets
from analyzer.expr_parser import parse_expr
import support


Number = Union[int, float]
ListNumber = List[Number]
Matrix = List[ListNumber]


class Variable:
    # TODO: документация
    def __init__(self, name: str, idx: int,
                 tao: Optional[int] = None, memory: int = 0, values: ListNumber = None) -> None:
        self._name = name
        self._idx = idx
        self._tao = tao
        self._memory = memory
        self._values = values

    # TODO: возможно сделать через setter
    def add_value(self, val: Number) -> None:
        if (type(val) is not int) or (type(val) is not float):
            raise TypeError('Некорректный тип. Ожидается int или float. type(val) = ' + str(type(val)))
        if len(self._values) < self._memory:
            self._values.append(val)
        elif len(self._values) == self._memory:
            self._values.pop(0)
            self._values.append(val)

    def initialization(self, values: ListNumber) -> None:
        self._memory = len(values)
        self._values = values  # .copy()

    def update(self, new_val: Number) -> None:
        self._values[:-1] = self._values[1:]
        self._values[-1] = new_val

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        return self._idx

    @property
    def tao(self) -> Optional[int]:
        return self._tao

    @property
    def memory(self) -> int:
        return self._memory

    @property
    def values(self) -> ListNumber:
        return self._values

    @values.setter
    def values(self, val) -> None:  # TODO: типы
        self._values = val

    @property
    def last_value(self) -> Number:
        return self._values[-1]


ListVars = List[Variable]


class GroupVariable:
    def __init__(self, list_var, min_memory=0):
        self._vars = sorted(list_var, key=operator.attrgetter('_tao'))
        self._max_tao = self._vars[-1].tao
        self._min_memory = min_memory
        self._memory = min_memory + self._max_tao
        self._values = None
        self._group_name = self._vars[0].name.split('_')[0]

    def update(self, val):
        self._values[:-1] = self._values[1:]
        self._values[-1] = val

    def init_zeros(self, min_memory=0):
        if min_memory > 0:
            self.set_min_memory(min_memory)
        val = [0. for _ in range(self._memory)]
        self.init_values(val)

    def init_values(self, val, min_memory=0):
        len_val = len(val)
        if min_memory > 0:
            self.set_min_memory(min_memory)
        if self._min_memory > 0:
            # заполнить значения
            if self._memory > len_val >= self._min_memory:
                # дополняем нулями
                self._values = val.copy() + [0. for _ in range(self._memory - len_val)]
                for v in self._vars:
                    v.values = self._values
            elif len_val == self._memory:
                self._values = val.copy()
                for v in self._vars:
                    v.values = self._values
            else:
                raise ValueError(f'Длнина списка должна быть >= {self._min_memory} и <= {self._memory}.')
        else:
            raise ValueError('Не установлен минимальный размер памяти.')

    def set_min_memory(self, val):
        self._memory = val + self._max_tao
        self._min_memory = val

    @property
    def last_value(self):
        return self._values[-1]

    @property
    def memory(self):
        return self._memory

    # @memory.setter
    # def memory(self, value):
    #     self._memory = value + self._min_memory
    #
    #     self._min_memory = value - self._max_tao
    #     if self._min_memory > 0:
    #         self._memory = value
    #     else:
    #         raise ValueError('некорректное значение memory')

    @property
    def variables(self):
        return self._vars

    @property
    def values(self):
        return self._values

    @property
    def group_name(self):
        return self._group_name

    @property
    def max_tao(self):
        return self._max_tao


class Model:
    # TODO: документация
    def __init__(self):
        self._expr_str = ''
        self._model_expr_str = ''

        self._x = []  # список экземпляров класса Variable
        self._u = []
        self._a = []  # список экземпляров класса Variable, у a tao = None всегда

        # функции
        self._grad = []
        self._func_model = None

        # self._func_u = None  # ???
        # self._func_obj = None  # ???

        # sympy выражения
        self._model_expr = None
        self._sp_var = []

    def initialization(self, a: Matrix, x: Matrix, u: Matrix, type_memory='min') -> Optional[NoReturn]:
        if a:
            self.variable_init(a, t='a', type_memory=type_memory)
        if x:
            self.variable_init(x, t='x', type_memory=type_memory)
        if u:
            self.variable_init(u, t='u', type_memory=type_memory)

    def variable_init(self, values: Matrix, t='a', type_memory='min') -> Optional[NoReturn]:
        if t in ['a', 'x', 'u']:
            attr = self.__getattribute__('_' + t)
            if not attr:
                raise ValueError('Не задан атрибут: ' + '_' + t)
            if (not values) or (len(values) != len(attr)):
                raise ValueError('len(values) != len(self._' + t + '): ' + str(len(values)) + ' != ' + str(len(attr)))
            for i in range(len(attr)):
                if not isinstance(values[i], list):
                    raise TypeError('Ожидается список')
                if t == 'a' and support.is_rect_matrix(values, sub_len=len(values[0])):
                    attr[i].initialization(values[i])
                elif t in ['x', 'u'] and support.is_rect_matrix(values, min_len=len(self._a)):
                    memory = len(self._a)
                    if type_memory == 'min':
                        attr[i].init_values(values[i], min_memory=memory)
                    elif type_memory == 'max':
                        if len(values[i]) >= (memory + attr[i].max_tao):
                            attr[i].set_min_memory(memory)
                            attr[i].init_values(values[i])
                        else:
                            raise ValueError(f'Количество памяти должно быть >= {memory + attr[i].max_tao}')
        else:
            raise ValueError('t = ' + t + '. t not in ["a", "x", "u"]')

    def create_variables(self, data: Union[list, dict], t: str) -> Optional[NoReturn]:
        variables = []
        if isinstance(data, list):
            for item in data:
                v = Variable(*item)
                variables.append(v)
        elif isinstance(data, dict):
            for key in data.keys():
                v = Variable(*[key, *data[key]])
                variables.append(v)
        name_attr = '_' + t
        if name_attr in self.__dict__:
            if t == 'a':
                variables.sort(key=operator.attrgetter('_idx'))
            else:
                variables = [
                    GroupVariable(list(g)) for k, g in itertools.groupby(sorted(variables,
                                                                                key=operator.attrgetter('_idx')),
                                                                         operator.attrgetter('_idx'))
                ]
            self.__setattr__(name_attr, variables)
        else:
            raise ValueError('Не сущестует атрибута с именем "' + name_attr + '"')

    def generate_func_grad(self) -> Optional[NoReturn]:
        if not self._sp_var:
            raise ValueError('Не сгенерированы sympy переменные')
        for c in self._a:
            self._grad.append(ufuncify(self._sp_var, self._model_expr.diff(c.name)))

    def generate_sp_var(self) -> None:
        self._sp_var = sp.var([v.name for v in [*support.flatten([g.variables for g in self._x]),
                                                *support.flatten([g.variables for g in self._u]), *self._a]])

    def generate_model_func(self) -> None:
        self._func_model = ufuncify(self._sp_var, self._model_expr)

    def update_a(self, a) -> None:
        if not self._a:
            raise AttributeError(f'Атрибут _a не задан')
        len_a = len(self._a)
        if len(a) != len_a:
            raise ValueError(f'Длина массива {a} должна быть = {len_a}. {len_a} != {len(a)}')
        for i in range(len_a):
            self._a[i].update(a[i])

    def update_x(self, val) -> None:
        if not self._x:
            raise AttributeError(f'Атрибут _x не задан')
        len_x = len(self._x)
        if len_x != len(val):
            raise ValueError(f'Длина массива {val} должна быть = {len_x}. {len_x} != {len(val)}')
        for i in range(len_x):  # группы
            self._x[i].update(val[i])

    def update_u(self, val) -> None:
        if not self._u:
            raise AttributeError(f'Атрибут _u не задан')
        len_u = len(self._u)
        if len_u != len(val):
            raise ValueError(f'Длина массива {val} должна быть = {len_u}. {len_u} != {len(val)}')
        for i in range(len_u):  # группы
            self._u[i].update(val[i])

    @property
    def expr_str(self) -> str:
        return self._expr_str

    @expr_str.setter
    def expr_str(self, val: str) -> None:
        self._expr_str = val

    @property
    def model_expr_str(self) -> str:
        return self._model_expr_str

    @model_expr_str.setter
    def model_expr_str(self, val: str) -> None:
        self._model_expr_str = val
        self._model_expr = sp.S(val)

    @property
    def model_expr(self) -> Add:  # ПРОВЕРИТЬ
        return self._model_expr

    @property
    def inputs(self):  # изменить тип
        return self._u

    @property
    def outputs(self):  # изменить тип
        return self._x

    @property
    def coefficients(self) -> ListVars:
        return self._a

    @property
    def a_values(self):
        return [a.values for a in self._a]

    @property
    def x_values(self):
        return [g.values for g in self._x]

    @property
    def u_values(self):
        return [g.values for g in self._u]

    @property
    def last_a(self):
        return [c.last_value for c in self._a]

    @property
    def last_x(self):
        return [group.last_value for group in self._x]

    @property
    def last_u(self):
        return [group.last_value for group in self._u]

    @property
    def grad(self):  # не ясен тип
        return self._grad

    @property
    def func_model(self):
        return self._func_model


def create_model(expr: str) -> Model:
    # TODO: документация
    if check_brackets(expr, brackets='()') != -1:
        raise ValueError('Некорректно расставлены скобки')
    model_expr, x_names, u_names, a_names = parse_expr(expr)
    model = Model()
    model.expr_str = expr
    model.model_expr_str = model_expr  # строковое и sympy выражения сохранены
    model.create_variables(x_names, t='x')
    model.create_variables(u_names, t='u')
    model.create_variables(a_names, t='a')  # переменные сгенерированы
    model.generate_sp_var()  # sympy переменные сгенерированы
    model.generate_func_grad()  # градиенты посчитаны
    model.generate_model_func()  # сгенерирована функция расчета значения модели

    return model


def main():
    model = create_model('a_0*x1(t-1)+a_3*x2(t-3)+a_2*x2(t-1)+a_1*x1(t-2)')

    print(model.model_expr_str)
    print(model.model_expr)

    print('-' * 20)
    for g in model.outputs:
        print('group:', g.group_name)
        for i in g.variables:
            print(i.name, i.tao)
    print('-' * 20)

    print(model.inputs)
    print(model.outputs)
    print(model.coefficients)
    print('-' * 20)

    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]
    x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    u = []
    model.initialization(a, x, u, type_memory='min')
    print(model.a_values)
    print(model.x_values)
    print(model.u_values)
    print('-' * 20)
    model.update_a([10, 20, 30, 40])
    print('a', model.a_values)
    model.update_x([50, 10])
    print('x1', model.x_values)
    model.update_x([90, 70])
    print('x2', model.x_values)
    print('-' * 20)
    print(model.last_a)
    print(model.last_x)
    print(model.last_u)
    print('-' * 20)


if __name__ == '__main__':
    main()
