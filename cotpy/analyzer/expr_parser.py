from typing import Tuple, Union, NoReturn, List

# TODO: пока только односимвольные имена
from cotpy.settings import variable_names, default_params


OPERATORS = ('+', '-', '*', '/', '**')
BRAKETS = '()'


def parse_expr(expr: str) -> Tuple[str, dict, dict, List[Tuple[str, int]]]:
    expr = expr.replace(' ', '')
    i = 0
    res = ''
    # TODO: возможно переделать x_names и u_names в списки tuples
    x_names: dict = {}  # ключи - индексы, значения - списки с именами или ключ - индекс, значение - (имя, тао)
    u_names: dict = {}  # ключи - индексы, значения - списки с именами
    a_names: list = []  #

    while i < len(expr):
        s = expr[i]
        name = ''
        if s == variable_names['obj']:
            name, idx, tao, i = parse_var(expr, i)
            if name not in x_names.keys():
                x_names[name] = (idx, tao)
        elif s == variable_names['control']:
            name, idx, tao, i = parse_var(expr, i)
            if name not in u_names.keys():
                u_names[name] = (idx, tao)
        elif s == variable_names['coefficient']:
            name, idx, _, i = parse_var(expr, i)
            if name not in [item[0] for item in a_names]:
                a_names.append((name, idx))
        elif s == variable_names['error']:
            pass
        elif s == variable_names['unknown_impact']:
            pass
        else:
            res += s
            i += 1
        res += name

    return res, x_names, u_names, a_names


def parse_var(expr: str, i: int) -> Union[NoReturn, Tuple[str, int, int, int]]:
    var_control = variable_names['control']
    var_obj = variable_names['obj']
    var_coef = variable_names['coefficient']
    var_time = variable_names['time']
    default_idx: int = default_params['default_idx']
    delimiter: str = default_params['delimiter']

    s = expr[i]
    idx = -1
    i += 1
    while i < len(expr) and expr[i] != ')':
        if expr[i] == delimiter:
            i += 1
        elif expr[i].isdigit():
            idx, i = int_parse(expr, i)
        elif expr[i] in OPERATORS:
            if s == var_coef:
                if idx == -1:
                    idx = default_idx  # last_idx + 1
                name = s + str(idx)
                return name, idx, 0, i
            elif s in (var_control, var_obj):
                message = 'Некорректное указание временной задержки. Ожидается: "(". Текущий символ: ' + expr[i] + '.'
                raise ValueError(message)
        elif expr[i] == '(':
            i += 1
            if s in (var_control, var_obj):
                tao, i = parse_lag(expr, i, var_time)
                if idx != -1:
                    name = s + str(idx) + '_' + str(tao)  # tao + 1
                else:
                    name = s + str(default_idx) + '_' + str(tao)  # tao + 1
                    idx = default_idx
                return name, idx, tao, i
            elif s == var_coef:
                message = 'Указание временной задержки у коэффициентов не поддерживаеся.'
                raise ValueError(message)
        else:
            message = get_error_message(reality=expr[i], position=i)
            raise ValueError(message)

    if i >= len(expr) or expr[i] == ')':
        if s == var_coef:
            if idx == -1:
                idx = default_idx  # last_idx + 1
            name = s + str(idx)
            return name, idx, 0, i
        elif s in (var_control, var_obj):
            message = 'Ожидается временная задержка'
            raise ValueError(message)


def parse_lag(expr: str, i: int, var_time: str) -> Union[NoReturn, Tuple[int, int]]:
    tao = 0
    start = i
    while i <= len(expr) and expr[i] != ')':
        if i - start == 0:
            if expr[i] == var_time:
                i += 1
            else:
                message = get_error_message(expect=var_time, reality=expr[i], position=i)
                raise ValueError(message)
        elif i - start == 1:
            if expr[i] == '-':
                i += 1
            else:
                message = get_error_message(expect='"-"', reality=expr[i], position=i)
                raise ValueError(message)
        elif i - start == 2:
            if expr[i].isdigit():
                tao, i = int_parse(expr, i)
                break
            else:
                message = get_error_message(expect='величина задержки', reality=expr[i], position=i)
                raise ValueError(message)

    if expr[i] == ')':
        return tao, i + 1  # tao - 1
    message = get_error_message(expect='")"', reality=expr[i], position=i)
    raise ValueError(message)


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_token(expr: str):
    sub_str = ''
    for c in expr:
        if c.isalpha():
            if (not sub_str) or sub_str.isalpha():
                sub_str += c
            else:
                yield sub_str
                sub_str = c
        elif c.isdigit():
            if (not sub_str) or is_float(sub_str):
                sub_str += c
            else:
                yield sub_str
                sub_str = c
        elif c in '.':
            if not sub_str:
                yield c
            elif sub_str.isdigit():
                sub_str += c
            else:
                yield sub_str
                sub_str = c
        elif c in BRAKETS or c == '_':
            if sub_str:
                yield sub_str
            sub_str = c
        elif c in OPERATORS:
            if sub_str in OPERATORS:
                sub_str += c
                continue
            else:
                yield sub_str
                sub_str = c
        else:
            if sub_str:
                yield sub_str
                sub_str = ''
            yield c
    if sub_str:
        yield sub_str


def int_parse(expr: str, i: int) -> Tuple[int, int]:
    numb = ''
    while i < len(expr) and expr[i].isdigit():
        numb += expr[i]
        i += 1
    return int(numb), i


def get_error_message(**kwargs) -> str:
    if 'expect' in kwargs:
        message = 'Ожидается ' + kwargs['expect']
    else:
        message = 'Неожиданный символ'
    message += '. Текущий символ "' + kwargs['reality'] + '", позиция: ' + str(kwargs['position']) + '.'
    return message
