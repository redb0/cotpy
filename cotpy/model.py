import copy
import itertools
import operator

from cotpy import support
import sympy as sp
import numpy as np
from cotpy.settings import variable_names, expr_vars, control_law_vars
from sympy.core.add import Add
from sympy.utilities.autowrap import ufuncify
from typing import Union, List, Optional, NoReturn

from cotpy.analyzer.validation import check_brackets

Number = Union[int, float]
ListNumber = List[Number]
Matrix = List[ListNumber]


class Variable:
    # TODO: документация
    def __init__(self, name: str, idx: int,
                 tao: Optional[int] = None, memory: int = 0, values: ListNumber = None, var_type='') -> None:
        self._name = name  # TODO: храть имя типа a, x, xp отдельно

        self._idx = idx
        self._tao = tao
        self._memory = memory
        self._values = values

        self._type = var_type

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
            # if self.name[:len(variable_names['obj'])] == variable_names['obj']:
            #     self.name = variable_names['predicted_outputs'] + self.name[len(variable_names['obj']):]
            # elif self.name[:len(variable_names['control'])] == variable_names['control']:
            #     self.name = variable_names['predicted_inputs'] + self.name[len(variable_names['control']):]
            # elif self.name[:len(expr_vars['add_input'])] == expr_vars['add_input']:
            #     self.name = control_law_vars['predicted_add_input'] + self.name[len(expr_vars['add_input']):]

            if self._type in expr_vars.keys():
                self._name = control_law_vars['predicted_'+self._type]
            # if self._type == expr_vars['output']:
            #     self._name = control_law_vars['predicted_output']
            # elif self._type == expr_vars['input']:
            #     self._name = control_law_vars['predicted_input']
            # elif self._type == expr_vars['add_input']:
            #     self._name = control_law_vars['predicted_add_input']

        self.update_tao(tao)

    def copy_var(self, var):
        self._name = var.name
        self._tao = var.tao
        self._idx = var.index
        self._memory = var.memory
        self._values = copy.copy(var.values)

    def update_tao(self, new_tao):
        self._tao = new_tao
        # items = self._name.split('_')
        # items[-1] = str(self._tao)  # self._tao + 1
        # self._name = '_'.join(items)

    @property
    def name(self) -> str:
        # return self._name
        if self._tao is None:
            return f'{self._name}{self._idx}'
        else:
            return f'{self._name}{self._idx}_{abs(self._tao)}'

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

    def get_last_value(self) -> Number:
        if self._values is not None or self._values != []:
            return self._values[-1]
        raise ValueError('Попытка извлечь значение из пустого списка')

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
            amount_additive = self._add_memory - (self._min_memory - self._max_tao + 1)
            if amount_additive < 0:
                amount_additive = 0
            self._values = np.hstack(([0. for _ in range(amount_additive)], val))
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
        self._str_expr = ''
        self._v = dict()
        self._outputs = None

        # функции
        self._grad = []
        self._func_model = None

        # sympy выражения
        self._sp_expr = None
        self._sp_var = []

    def initialization(self, type_memory: str='max', memory_size: int=0, **kwargs) -> Optional[NoReturn]:
        kw = support.normalize_kwargs(kwargs, alias_map=expr_vars)
        for k, v in self._v.items():
            if v and k in kw:
                self.variable_init(k, kw[k], type_memory=type_memory, memory_size=memory_size)
            elif k == 'output' and memory_size > 0:
                if v is not None:
                    self._outputs = np.array(v[0])
                else:
                    self._outputs = np.array([0 for _ in range(memory_size)])
            else:
                self.variable_init(t=k, memory_size=memory_size)

    def variable_init(self, t, values: Matrix=None, type_memory: str='max', memory_size: int=0) -> Optional[NoReturn]:
        if t not in expr_vars.keys():
            raise ValueError(f't = {t}. t not in {expr_vars.keys()}')
        if (values is not None) and (len(values) != len(self._v[t])):
            raise ValueError(f'Передан некорректный массив, ключ - {repr(t)}. '
                             f'len(values) != len(self._{t}): {len(values)} != {len(self._v[t])}')

        memory = len(self._v['coefficient'])
        for i in range(len(self._v[t])):
            if values is None:
                if t == 'coefficient':
                    self._v[t][i].init_ones(memory)  # attr
                elif t in expr_vars.keys():
                    self._v[t][i].init_zeros(min_memory=memory)  # attr
            else:
                if not support.is_rect_matrix(values, min_len=memory):
                    raise ValueError(f'Длина подмассивов атрибута "{t}" должна быть >= {memory}')
                if t == 'coefficient':
                    self._v[t][i].initialization(values[i])  # attr
                elif t in expr_vars.keys():
                    if len(values[i]) < (memory + self._v[t][i].max_tao - 1):  # attr
                        raise ValueError(f'Количество элементов атребута "{t}" '
                                         f'должно быть >= {memory + self._v[t][i].max_tao - 1}')  # attr
                    if type_memory == 'min':
                        self._v[t][i].set_memory(memory)  # attr
                    elif type_memory == 'max':
                        max_memory = len(values[i]) - self._v[t][i].max_tao + 1  # attr
                        self._v[t][i].set_memory(max_memory)  # attr
                    if memory_size > 0:  # memory_size >= memory:
                        self._v[t][i].set_additional_memory(memory_size)  # attr
                    self._v[t][i].init_values(values[i])  # attr

    def create_var(self, d):
        for k, v in d.items():
            variables = []
            for name, data in v.items():
                variables.append(Variable(expr_vars[k], *data, var_type=k))  # Variable(name, *data, var_type=k)
            if k == 'coefficient':
                variables.sort(key=operator.attrgetter('_idx'))
            else:
                variables = [
                    GroupVariable(list(g)) for k, g in itertools.groupby(sorted(variables,
                                                                                key=operator.attrgetter('_idx')),
                                                                         operator.attrgetter('_idx'))
                ]
            self._v[k] = variables
            # print(variables)

    def generate_func_grad(self) -> Optional[NoReturn]:
        if not self._sp_var:
            raise ValueError('Не сгенерированы sympy переменные')
        for c in self._v['coefficient']:
            # print(self._sp_expr.diff(c.name))
            self._grad.append(ufuncify(self._sp_var, self._sp_expr.diff(c.name)))

    def generate_sp_var(self) -> None:
        names = []
        for item in self._v.values():
            if item:
                if isinstance(item[0], GroupVariable):
                    names += [k.name for k in list(support.flatten([g.variables for g in item]))]
                else:
                    names += [k.name for k in item]
        self._sp_var = sp.var(names)

    def get_var_values(self, t: str, n: int=0, is_new: bool=True):
        tmp_expr_vars = {v: k for k, v in expr_vars.items()}
        if t in expr_vars.values():
            t = tmp_expr_vars[t]
        elif t not in self._v.keys() or t == 'coefficient':
            raise ValueError(f'Аргумент t должен принимать одно из значений: '
                             f'{repr(k) for k in self._v.keys() if k != "coefficient"}')
        return [g.all_tao_values(n=n, is_new=is_new) for g in self._v[t]]

    def get_last_model_value(self) -> Number:  # TODO: Убрать и сделать один метод get_model_value
        return self._func_model(*list(support.flatten(self.get_var_last_value().values())))

    def get_value(self, a: Optional[ListNumber]) -> Number:
        if a is None:
            return self.get_last_model_value()
        else:
            val = self.get_var_last_value()
            val['coefficient'] = a
            return self._func_model(*list(support.flatten(val.values())))

    def generate_model_func(self) -> None:
        self._func_model = ufuncify(self._sp_var, self._sp_expr)

    def get_grad_value(self, *args, **kwargs) -> ListNumber:
        if args:
            return [f(*args) for f in self._grad]
        elif kwargs:
            kw = support.normalize_kwargs(kwargs, alias_map=expr_vars)
            val = self.get_var_last_value()
            for k in val.keys():
                if k in kw:
                    val[k] = kw[k]
            return [f(*list(support.flatten(val.values()))) for f in self._grad]
        else:
            return [f(*list(support.flatten(self.get_var_last_value().values()))) for f in self._grad]

    def update_data(self, **kwargs) -> Optional[NoReturn]:
        if not kwargs:
            return
        kw = support.normalize_kwargs(kwargs, alias_map=expr_vars)
        for k, v in self._v.items():
            if k == 'output' and k in kw:
                self.update_x(kw['output'])
            if v and k in kw:
                self.update_var_value(kw[k], var=k)

    def update_var_value(self, value, var: str):
        if var not in self._v.keys():
            raise ValueError(f'Аргумент value может принимать одно из значений: {repr(k) for k in self._v.keys()}')
        l = len(self._v[var])
        if len(value) != l:
            raise ValueError(f'Длина массива value должна быть = {l}')
        for i in range(l):
            self._v[var][i].update(value[i])

    def update_a(self, value: ListNumber) -> None:
        self.update_var_value(value, var='coefficient')
        # len_a = len(self._v['coefficient'])
        # if len(a) != len_a:
        #     raise ValueError(f'Длина массива {a} должна быть = {len_a}. {len_a} != {len(a)}')
        # for i in range(len_a):
        #     self._v['coefficient'][i].update(a[i])

    def update_x(self, value: ListNumber) -> None:
        if self._v['output']:
            self.update_var_value(value, var='output')
            # len_x = len(self._v['output'])
            # if len_x != len(val):
            #     raise ValueError(f'Длина массива {val} должна быть = {len_x}. {len_x} != {len(val)}')
            # for i in range(len_x):  # группы
            #     self._v['output'][i].update(val[i])
        else:
            self._outputs[:-1] = self._outputs[1:]
            self._outputs[-1] = value[0]

    def update_u(self, value: ListNumber) -> None:
        self.update_var_value(value, var='input')
        # len_u = len(self._v['input'])
        # if len_u != len(val):
        #     raise ValueError(f'Длина массива {val} должна быть = {len_u}. {len_u} != {len(val)}')
        # for i in range(len_u):  # группы
        #     self._v['input'][i].update(val[i])

    def update_z(self, value: ListNumber) -> None:
        self.update_var_value(value, var='add_input')

    def set_add_memory(self, n: int) -> None:
        # TODO: возможно сделать изменение памяти без удаления существующих значений
        if n > 0:
            if self._v['output']:
                for group in self._v['output']:
                    group.set_additional_memory(n)
            else:
                self._outputs = np.array([0 for _ in range(n)])
            if self._v['input']:
                for group in self._v['input']:
                    group.set_additional_memory(n)
        else:
            raise ValueError(f'Величина памяти должна быть >= 0')

    def get_var_last_value(self):
        res = dict()
        for k, v in self._v.items():
            if v:
                if isinstance(v[0], GroupVariable):
                    res[k] = [g.all_tao_values(0, is_new=True) for g in self._v[k]]
                else:
                    res[k] = [a.get_last_value() for a in v]
        return res

    def get_all_var_values(self):
        res = dict()
        for k, v in self._v.items():
            if v:
                res[k] = [a.values for a in v]
            else:
                res[k] = []
        return res

    @property
    def memory_size(self) -> int:
        if self._v['output']:
            return self._v['output'][0].memory_size
        elif self._outputs is not None:
            return len(self._outputs)
        else:
            return 0

    @property
    def start_memory_size(self) -> int:
        return self._v['output'][0].start_memory_size

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
        return self._v['input']

    @property
    def outputs(self) -> List[GroupVariable]:
        return self._v['output']

    def outputs_values(self, p: int):
        if self._v['output']:
            return [list(support.flatten(self.get_var_values(t='output', n=-i)))[0] for i in range(p)]
        else:
            # print('получение выходов:', self._outputs[-p:])
            return self._outputs[-p:]

    @property
    def coefficients(self) -> ListVars:
        return self._v['coefficient']

    # @property
    # def a_values(self) -> List[ListNumber]:
    #     return [a.values for a in self._v['coefficient']]

    # @property
    # def x_values(self) -> List[ListNumber]:
    #     return [g.values for g in self._v['output']]

    # @property
    # def u_values(self) -> List[ListNumber]:
    #     return [g.values for g in self._v['input']]

    @property
    def last_a(self) -> ListNumber:
        return [c.get_last_value() for c in self._v['coefficient']]

    @property
    def last_x(self) -> ListNumber:
        return [group.last_value for group in self._v['output']]  # get_last_value()

    @property
    def last_u(self) -> ListNumber:
        return [group.last_value for group in self._v['input']]  # get_last_value()

    def get_outputs_value(self, i: int):
        if not self._v['output']:
            return []
        if i < len(self._v['coefficient']):
            return [group.values[i] for group in self._v['output']]
        else:
            pass

    def get_inputs_value(self, i: int):
        if not self._v['input']:
            return []
        if i < len(self._v['coefficient']):
            return [group.values[i] for group in self._v['input']]
        else:
            pass

    def get_coefficients_value(self, i: int):
        if not self._v['coefficient']:
            return []
        if i < len(self._v['coefficient']):
            return [c.values[i] for c in self._v['coefficient']]
        else:
            pass

    @property
    def model_vars(self):
        return self._v

    @property
    def grad(self):  # list[numpy.ufunc]
        return self._grad

    @property
    def func_model(self):  # numpy.ufunc
        return self._func_model

    @property
    def sp_var(self):
        return self._sp_var

    def __repr__(self):
        pass

    def __str__(self) -> str:
        return f'Model("{self._sp_expr}")'


def create_model(expr: str) -> Model:
    # TODO: документация
    if not check_brackets(expr, brackets='()'):
        raise ValueError('Некорректно расставлены скобки')
    from cotpy.de_parser.parser import expr_parse
    model_expr, model_vars = expr_parse(expr)
    model = Model()
    model.str_expr = model_expr  # строковое и sympy выражения сохранены
    model.create_var(model_vars)
    model.generate_sp_var()  # sympy переменные сгенерированы
    model.generate_func_grad()  # градиенты посчитаны
    model.generate_model_func()  # сгенерирована функция расчета значения модели

    return model


def main():
    expr = "a0+a1*x(t-1)+a2*x(t-2)+a3*u(t-6)+a4*u(t-7)+a5*z1(t-1)+a6*z2(t-1)"
    m = create_model(expr)
    a = [[1, 1, 1, 1, 1, 0, 5], [1, 1, 1, 1, 1, 1, 6],
         [1, 1, 1, 1, 1, 2, 7], [1, 1, 1, 1, 1, 3, 8],
         [1, 1, 1, 1, 1, 4, 9], [1, 1, 1, 1, 1, 3, 10], [1, 1, 1, 1, 1, 2, 11]]

    m.initialization(type_memory='max', memory_size=0, a=a)

    print(m.get_all_var_values())

if __name__ == '__main__':
    main()
