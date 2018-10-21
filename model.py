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

    def initialization(self, a: Matrix, x: Matrix, u: Matrix) -> Optional[NoReturn]:
        self.variable_init(a, t='a')
        self.variable_init(x, t='x')
        self.variable_init(u, t='u')

    def variable_init(self, values: Matrix, t='a') -> Optional[NoReturn]:
        if t in ['a', 'x', 'u']:
            attr = self.__getattribute__('_' + t)
            if not attr:
                raise ValueError('Не задан атрибут: ' + '_' + t)
            if (not values) or (len(values) != len(attr)):
                raise ValueError('len(values) != len(self._' + t + '): ' + str(len(values)) + ' != ' + str(len(attr)))
            memory = 0
            for i in range(len(attr)):
                if not isinstance(values[i], list):
                    raise TypeError('Ожидается список')
                if i == 0:
                    memory = len(values[0])
                if memory == len(values[i]):
                    attr[i].initialization(values[i])
                else:
                    message = 'Не совпадает кол-во элементов. ' + str(memory) + ' != ' + str(len(values[i]))
                    raise ValueError(message)
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
                    list(g) for k, g in itertools.groupby(sorted(variables, key=operator.attrgetter('_idx', '_tao')),
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
        self._sp_var = sp.var([v.name for v in [*support.flatten(self._x), *support.flatten(self._u), *self._a]])

    def generate_model_func(self) -> None:
        self._func_model = ufuncify(self._sp_var, self._model_expr)

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
    def inputs(self) -> ListVars:
        return self._u

    @property
    def outputs(self) -> ListVars:
        return self._x

    @property
    def coefficients(self) -> ListVars:
        return self._a

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
    print(x_names)
    print(u_names)
    print(a_names)
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
        print('group:')
        for i in g:
            print(i.name, i.tao)
    print('-' * 20)

    print(model.inputs)
    print(model.outputs)
    print(model.coefficients)

    # a = [2, 3, 4]
    # x = [2, 3, 4]
    # print(model.func_model(*x, *a))


if __name__ == '__main__':
    main()
