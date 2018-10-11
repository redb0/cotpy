from typing import Union, List

from sympy.utilities.autowrap import ufuncify
import sympy as sp

from exprparse.validation import check_brackets
from exprparse.expr_parser import parse_expr


Number = Union[int, float]
ListNumber = List[Union[int, float]]


class Variable:
    # TODO: документация
    def __init__(self, name: str, idx: int,
                 tao: Union[int, None] = None, memory: int = 0, values: ListNumber = None) -> None:
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
        self._values = values.copy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        return self._idx

    @property
    def tao(self) -> Union[None, int]:
        return self._tao

    @property
    def memory(self) -> int:
        return self._memory

    @property
    def values(self) -> ListNumber:
        return self._values

    @property
    def last_value(self):  # добавить тип
        return self._values[-1]


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

    def initialization(self, a, x, u) -> None:  # добавить тип
        self.variable_init(a, t='a')
        self.variable_init(x, t='x')
        self.variable_init(u, t='u')

    def variable_init(self, values, t='a'):  # добавить тип
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

    def create_variables(self, data, t: str):  # добавить тип
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
            self.__setattr__(name_attr, variables)
        else:
            raise ValueError('Не сущестует атрибута с именем "' + name_attr + '"')

    def generate_func_grad(self):  # добавить тип
        if not self._sp_var:
            raise ValueError('Не сгенерированы sympy переменные')
        for c in self._a:
            self._grad.append(ufuncify(self._sp_var, self._model_expr.diff(c.name)))

    def generate_sp_var(self):  # добавить тип
        self._sp_var = sp.var([v.name for v in [*self._x, *self._u, *self._a]])

    def generate_model_func(self):  # добавить тип
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
    def model_expr(self):  # добавить тип
        return self._model_expr

    @property
    def inputs(self):  # добавить тип
        return self._u

    @property
    def outputs(self):  # добавить тип
        return self._x

    @property
    def coefficients(self):  # добавить тип
        return self._a

    @property
    def grad(self):  # добавить тип
        return self._grad

    @property
    def func_model(self):  # добавить тип
        return self._func_model


def create_model(expr: str) -> Model:
    # TODO: документация
    if check_brackets(expr, brackets='()') != -1:
        raise ValueError('Некорректно расставлены скобки')
    model_expr, x_names, u_names, a_names = parse_expr(expr)
    model = Model()
    model.expr = expr
    model.model_expr_str = model_expr  # строковое и sympy выражения сохранены
    model.create_variables(x_names, t='x')
    model.create_variables(u_names, t='u')
    model.create_variables(a_names, t='a')  # переменные сгенерированы
    model.generate_sp_var()  # sympy переменные сгенерированы
    model.generate_func_grad()  # градиенты посчитаны
    model.generate_model_func()  # сгенерирована функция расчета значения модели

    return model


def main():
    model = create_model('a_0*x(t-1)+a+2*a+cos(u(t-1))+a_1*u(t-2)')
    # model = create_model('a_0*x(t-1)')

    print(model.model_expr_str)
    print(model.model_expr)

    print(model.inputs)
    print(model.outputs)
    print(model.coefficients)


if __name__ == '__main__':
    main()
