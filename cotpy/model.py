import copy
import itertools
import operator

from cotpy import support
import sympy as sp
import numpy as np
from cotpy.settings import variable_names
from sympy.core.add import Add
from sympy.utilities.autowrap import ufuncify
from typing import Union, List, Optional, NoReturn

from cotpy.analyzer.validation import check_brackets
from cotpy.analyzer.expr_parser import parse_expr

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
            raise TypeError(f'Некорректный тип. Ожидается int или float. type(val) = {type(val)}')
        if len(self._values) < self._memory:
            self._values.append(val)
        elif len(self._values) == self._memory:
            self._values.pop(0)
            self._values.append(val)

    def initialization(self, values: ListNumber) -> None:
        self._memory = len(values)
        self._values = values  # .copy()

    def init_zeros(self, n: int) -> None:
        self._memory = n
        self._values = np.zeros((n, ))

    def init_ones(self, n: int) -> None:
        self._memory = n
        self._values = np.ones((n, ))

    def update(self, new_val: Number) -> Optional[NoReturn]:
        if self._values is None or self._values == []:
            raise ValueError('Атрибут _values не задан')
        if isinstance(new_val, (int, float, np.number)):
            self._values[:-1] = self._values[1:]
            self._values[-1] = new_val
        else:
            raise ValueError('Некорректный тип данных, ожидается: int, float')

    def get_value(self):
        return self._values[-self._tao]

    def get_tex_name(self):
        if self.name[:len(variable_names['obj'])] == variable_names['obj']:
            if self.tao == 0:
                return f'{variable_names["obj"]}_{{{self._idx}}}({variable_names["time"]})'
            return f'{variable_names["obj"]}_{{{self._idx}}}({variable_names["time"]} - {self.tao})'
        elif self.name[:len(variable_names['control'])] == variable_names['control']:
            if self.tao == 0:
                return f'{variable_names["control"]}_{{{self._idx}}}({variable_names["time"]})'
            return f'{variable_names["control"]}_{{{self._idx}}}({variable_names["time"]} - {self.tao})'
        elif self.name[:len(variable_names['predicted_outputs'])] == variable_names['predicted_outputs']:
            return f'{variable_names["obj"]}_{{{self._idx}}}({variable_names["time"]} + {self.tao})'
        elif self.name[:len(variable_names['predicted_inputs'])] == variable_names['predicted_inputs']:
            return f'{variable_names["control"]}_{{{self._idx}}}({variable_names["time"]} + {self.tao})'
        elif self.name[:len(variable_names['coefficient'])] == variable_names['coefficient']:
            return f'{variable_names["coefficient"]}_{{{self._idx}}}({variable_names["time"]} + {self.tao})'

    def update_name_tao(self, tao):  # tao=var.tao - 1
        if tao < 0:
            if self.name[:len(variable_names['obj'])] == variable_names['obj']:
                self.name = variable_names['predicted_outputs'] + self.name[len(variable_names['obj']):]
            elif self.name[:len(variable_names['control'])] == variable_names['control']:
                self.name = variable_names['predicted_inputs'] + self.name[len(variable_names['control']):]
        self.update_tao(tao)

    def copy_var(self, var):
        self._name = var.name
        self._tao = var.tao
        self._idx = var.index
        self._memory = var.memory
        self._values = copy.copy(var.values)

    def update_tao(self, new_tao):
        self._tao = abs(new_tao)
        items = self._name.split('_')
        items[-1] = str(self._tao)  # self._tao + 1
        self._name = '_'.join(items)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

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
    def values(self, val) -> None:
        self._values = val

    @property
    def last_value(self) -> Number:
        return self._values[-1]

    def __repr__(self) -> str:
        return f'Variable({repr(self._name)}, {self._idx}, {self._tao}, {self._memory}, {self._values})'

    def __str__(self) -> str:
        return (f'Variable({repr(self._name)}, {self._idx}, '
                f'tao={self._tao}, memory={self._memory}, values={self._values})')


ListVars = List[Variable]


class GroupVariable:
    def __init__(self, list_var: ListVars) -> None:
        self._vars: ListVars = sorted(list_var, key=operator.attrgetter('_tao'))
        self._max_tao: int = self._vars[-1].tao
        # self._min_memory: int = min_memory
        self._min_memory: int = self._max_tao
        self._add_memory = 0
        self._values = None
        self._group_name: str = self._vars[0].name.split('_')[0]

    def update(self, val):
        self._values[:-1] = self._values[1:]
        self._values[-1] = val

    def add_vars(self, var):
        if isinstance(var, list):
            self._vars.extend(var)
        else:
            self._vars.append(var)
        self.sorted_var()
        # self._vars = sorted(self._vars, key=operator.attrgetter('_tao'))
        # self._max_tao: int = self._vars[-1].tao

    def add_unique_var(self, src):
        if isinstance(src, list):
            for v in src:
                if v.name not in [var.name for var in self._vars]:
                    self.add_vars(copy.copy(v))
        else:
            if src.name not in [var.name for var in self._vars]:
                self.add_vars(copy.copy(src))

    def sorted_var(self):
        self._vars = sorted(self._vars, key=operator.attrgetter('_tao'))
        self._max_tao: int = self._vars[-1].tao

    def init_zeros(self, min_memory: int=0) -> None:
        self.set_memory(min_memory)
        val = np.zeros((self._min_memory,))
        self.init_values(val)

    def init_ones(self, min_memory: int=0) -> None:
        self.set_memory(min_memory)
        val = np.ones((self._min_memory,))
        self.init_values(val)

    def init_values(self, val):
        len_val = len(val)
        if self._min_memory == 0:
            raise ValueError('Не установлен минимальный размер памяти.')
            # заполнить значения
        if len_val >= self._min_memory:
            self._values = np.array([0. for _ in range(self._add_memory - (self._min_memory - self._max_tao + 1))] + val)
            for v in self._vars:
                v.values = self._values
        else:
            raise ValueError(f'Длнина списка должна быть >= {self._min_memory}.')

    def set_additional_memory(self, n: int) -> None:
        memory = self._min_memory - self._max_tao + 1
        if n >= 0:  # memory:
            if self._values is None:
                self._add_memory = n
            else:
                self._add_memory = n
                self._values = np.concatenate((np.array([0. for _ in range(n - memory)]),
                                               self._values[-self._min_memory:]), axis=0)
                for v in self._vars:
                    v.values = self._values
        else:
            raise ValueError(f'Величина памяти должна быть >= 0')

    def set_memory(self, val) -> None:
        if val == 0:
            val = 1
        self._min_memory = val + self._max_tao - 1

    @property
    def memory_size(self) -> int:
        return self._add_memory

    @property
    def start_memory_size(self) -> int:
        return self._min_memory - self._max_tao + 1

    @property
    def last_value(self):
        return self._values[-1]

    def replace(self, src: Union[Variable, List[Variable]]):
        if isinstance(src, list):
            for v in src:
                if v.name not in [var.name for var in self._vars]:
                    self._vars.append(copy.copy(v))
        else:
            if src.name not in [var.name for var in self._vars]:
                self._vars.append(copy.copy(src))

    def all_tao_values(self, n: int=0, is_new: bool=True):  # -> ListNumber
        # хранятся в порядке убывания tao, self._memory = mem + tao - 1
        if n <= 0:
            return [self._values[n - v.tao] for v in self._vars]
        else:
            n -= 1
            if is_new:
                if n >= self._min_memory - self._max_tao + 1:
                    raise IndexError(f'Индекс за пределами диапозона. n >= {self._min_memory - self._max_tao + 1}')
                return [self._values[n - self._min_memory + self._max_tao - v.tao] for v in self._vars]
            if n > len(self._values) - self._max_tao:
                raise IndexError(f'Индекс за пределами диапозона. n >= {len(self._values) - self._max_tao}')
            else:
                return [self._values[n + self._max_tao - v.tao] for v in self._vars]

    @property
    def memory(self) -> int:
        return self._min_memory

    @property
    def variables(self):
        return self._vars

    @property
    def values(self):
        return self._values

    @property
    def group_name(self) -> str:
        return self._group_name

    @property
    def max_tao(self) -> int:
        return self._max_tao

    @property
    def min_tao(self) -> int:
        return min([v.tao for v in self._vars])

    def __repr__(self) -> str:
        return f'GroupVariable({repr(self._vars)})'

    def __str__(self) -> str:
        return (f'(_vars={self._vars}, _max_tao={self._max_tao},'
                f'_memory={self._min_memory}, _values={self._values}, _group_name={self._group_name})')


class Model:
    # TODO: документация
    def __init__(self):
        # self._expr_str = ''
        self._str_expr = ''
        self._x = []  # список экземпляров класса GroupVariable
        self._u = []
        self._a = []  # список экземпляров класса Variable, у a tao = None всегда

        self._outputs = None

        # функции
        self._grad = []
        self._func_model = None

        # sympy выражения
        self._sp_expr = None
        self._sp_var = []

    def initialization(self, a: Matrix=None, x: Matrix=None, u: Matrix=None,
                       type_memory: str='max', memory_size: int=0) -> Optional[NoReturn]:
        if self._a:
            if a is not None:
                self.variable_init(a, t='a', type_memory=type_memory)
            else:
                self.variable_init(t='a')
        if self._x:
            if x is not None:
                self.variable_init(x, t='x', type_memory=type_memory, memory_size=memory_size)
            else:
                self.variable_init(t='x', memory_size=memory_size)
        else:
            # создание дополнительного массива при отсутствии массива с иксами
            if memory_size > 0:
                if x is not None:
                    self._outputs = np.array(x[0])
                else:
                    self._outputs = np.array([0 for _ in range(memory_size)])
        if self._u:
            if u is not None:
                self.variable_init(u, t='u', type_memory=type_memory, memory_size=memory_size)
            else:
                self.variable_init(t='u', memory_size=memory_size)

    def variable_init(self, values: Matrix=None, t: str='a', type_memory: str='max', memory_size: int=0) -> Optional[NoReturn]:
        if t not in ['a', 'x', 'u']:
            raise ValueError(f't = {t}. t not in ["a", "x", "u"]')
        attr = self.__getattribute__('_' + t)
        if not attr:
            raise ValueError(f'Не задан атрибут: _{t}')
        if values and (len(values) != len(attr)):
            raise ValueError(f'Передан некорректный массив. '
                             f'len(values) != len(self._{t}): {len(values)} != {len(attr)}')

        memory = len(self._a)
        for i in range(len(attr)):
            if values is None:
                if t == 'a':
                    attr[i].init_ones(memory)
                elif t in ['x', 'u']:
                    attr[i].init_ones(min_memory=memory)
            else:
                if not support.is_rect_matrix(values, min_len=memory):
                    raise ValueError(f'Длина подмассивов атрибута "{t}" должна быть >= {memory}')
                if t == 'a':
                    attr[i].initialization(values[i])
                elif t in ['x', 'u']:
                    if len(values[i]) < (memory + attr[i].max_tao - 1):
                        raise ValueError(f'Количество элементов атребута "{t}" '
                                         f'должно быть >= {memory + attr[i].max_tao - 1}')
                    if type_memory == 'min':
                        attr[i].set_memory(memory)
                    elif type_memory == 'max':
                        max_memory = len(values[i]) - attr[i].max_tao + 1
                        attr[i].set_memory(max_memory)
                    if memory_size > 0:  # memory_size >= memory:
                        attr[i].set_additional_memory(memory_size)
                    attr[i].init_values(values[i])

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
            raise ValueError(f'Не сущестует атрибута с именем "{name_attr}"')

    def generate_func_grad(self) -> Optional[NoReturn]:
        if not self._sp_var:
            raise ValueError('Не сгенерированы sympy переменные')
        for c in self._a:
            # print(self._model_expr.diff(c.name))
            self._grad.append(ufuncify(self._sp_var, self._sp_expr.diff(c.name)))

    def generate_sp_var(self) -> None:
        self._sp_var = sp.var([v.name for v in [*list(support.flatten([g.variables for g in self._x])),
                                                *list(support.flatten([g.variables for g in self._u])), *self._a]])

    def get_x_values(self, n: int=0, is_new: bool=True) -> List[ListNumber]:
        return [g.all_tao_values(n, is_new=is_new) for g in self._x]

    def get_u_values(self, n: int=0, is_new: bool=True) -> List[ListNumber]:
        return [g.all_tao_values(n, is_new=is_new) for g in self._u]

    def get_last_model_value(self) -> Number:
        return self._func_model(*list(support.flatten(self.get_x_values())),
                                *list(support.flatten(self.get_u_values())),
                                *self.last_a)

    def get_value(self, a: Optional[ListNumber]) -> Number:
        if a is None:
            return self.get_last_model_value()
        else:
            return self._func_model(*list(support.flatten(self.get_x_values())),
                                    *list(support.flatten(self.get_u_values())),
                                    *a)

    def generate_model_func(self) -> None:
        self._func_model = ufuncify(self._sp_var, self._sp_expr)

    # def get_grad_value(self, x=(), u=(), a=()) -> ListNumber:
    #     return [f(*x, *u, *a) for f in self._grad]

    def get_grad_value(self, *args) -> ListNumber:
        return [f(*args) for f in self._grad]

    def get_last_grad_value(self):
        return self.get_grad_value(*list(support.flatten(self.get_x_values())),
                                   *list(support.flatten(self.get_u_values())),
                                   *self.last_a)

    def update_data(self, a: ListNumber=None, u: ListNumber=None, x: ListNumber=None) -> Optional[NoReturn]:
        if self._a:
            if a is not None:
                self.update_a(a)
            else:
                raise AttributeError(f'Требуется обновление коэффициентов. Передано значение: {a}')
        if self._x:
            if x is not None:
                self.update_x(x)
            else:
                raise AttributeError(f'Требуется обновление выходов. Передано значение: {x}')
        if self._u:
            if u is not None:
                self.update_u(u)
            else:
                raise AttributeError(f'Требуется обновление входов. Передано значение: {u}')

    def update_a(self, a: ListNumber) -> None:
        len_a = len(self._a)
        if len(a) != len_a:
            raise ValueError(f'Длина массива {a} должна быть = {len_a}. {len_a} != {len(a)}')
        for i in range(len_a):
            self._a[i].update(a[i])

    def update_x(self, val: ListNumber) -> None:
        if self._x:
            len_x = len(self._x)
            if len_x != len(val):
                raise ValueError(f'Длина массива {val} должна быть = {len_x}. {len_x} != {len(val)}')
            for i in range(len_x):  # группы
                self._x[i].update(val[i])
        else:
            self._outputs[:-1] = self._outputs[1:]
            self._outputs[-1] = val[0]

    def update_u(self, val: ListNumber) -> None:
        len_u = len(self._u)
        if len_u != len(val):
            raise ValueError(f'Длина массива {val} должна быть = {len_u}. {len_u} != {len(val)}')
        for i in range(len_u):  # группы
            self._u[i].update(val[i])

    def set_add_memory(self, n: int) -> None:
        # TODO: возможно сделать изменение памяти без удаления существующих значений
        if n > 0:
            if self._x:
                for group in self._x:
                    group.set_additional_memory(n)
            else:
                self._outputs = np.array([0 for _ in range(n)])
            if self._u:
                for group in self._u:
                    group.set_additional_memory(n)
        else:
            raise ValueError(f'Величина памяти должна быть >= 0')

    @property
    def memory_size(self) -> int:
        if self._x:
            return self._x[0].memory_size
        elif self._outputs is not None:
            return len(self._outputs)
        else:
            return 0

    @property
    def start_memory_size(self) -> int:
        return self._x[0].start_memory_size

    @property
    def str_expr(self) -> str:
        return self._str_expr

    @str_expr.setter
    def str_expr(self, val: str) -> None:
        self._str_expr = val
        self._sp_expr = sp.S(val)

    @property
    def sp_expr(self) -> Add:
        return self._sp_expr

    @property
    def inputs(self) -> List[GroupVariable]:
        return self._u

    @property
    def outputs(self) -> List[GroupVariable]:
        return self._x

    def outputs_values(self, p: int):
        if self._x:
            return [list(support.flatten(self.get_x_values(-i)))[0] for i in range(p)]
        else:
            # print('получение выходов:', self._outputs[-p:])
            return self._outputs[-p:]

    @property
    def coefficients(self) -> ListVars:
        return self._a

    @property
    def a_values(self) -> List[ListNumber]:
        return [a.values for a in self._a]

    @property
    def x_values(self) -> List[ListNumber]:
        return [g.values for g in self._x]

    @property
    def u_values(self) -> List[ListNumber]:
        return [g.values for g in self._u]

    @property
    def last_a(self) -> ListNumber:
        return [c.last_value for c in self._a]

    @property
    def last_x(self) -> ListNumber:
        return [group.last_value for group in self._x]

    @property
    def last_u(self) -> ListNumber:
        return [group.last_value for group in self._u]

    def get_outputs_value(self, i: int):
        if not self._x:
            return []
        if i < len(self._a):
            return [group.values[i] for group in self._x]
        else:
            pass

    def get_inputs_value(self, i: int):
        if not self._u:
            return []
        if i < len(self._a):
            return [group.values[i] for group in self._u]
        else:
            pass

    def get_coefficients_value(self, i: int):
        if not self._a:
            return []
        if i < len(self._a):
            return [c.values[i] for c in self._a]
        else:
            pass

    @property
    def grad(self):  # list[numpy.ufunc]
        return self._grad

    @property
    def func_model(self):  # numpy.ufunc
        return self._func_model

    def __repr__(self):
        pass

    def __str__(self) -> str:
        return f'Model("{self._sp_expr}")'


def create_model(expr: str) -> Model:
    # TODO: документация
    if not check_brackets(expr, brackets='()'):
        raise ValueError('Некорректно расставлены скобки')
    model_expr, x_names, u_names, a_names = parse_expr(expr)
    model = Model()
    # model.expr_str = expr
    model.str_expr = model_expr  # строковое и sympy выражения сохранены
    model.create_variables(x_names, t='x')
    model.create_variables(u_names, t='u')
    model.create_variables(a_names, t='a')  # переменные сгенерированы
    model.generate_sp_var()  # sympy переменные сгенерированы
    model.generate_func_grad()  # градиенты посчитаны
    model.generate_model_func()  # сгенерирована функция расчета значения модели

    return model
