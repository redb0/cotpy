from typing import Tuple, Union


# TODO: пока только односимвольные имена
variable_names = {
    'obj': 'x',
    'control': 'u',
    'coefficient': 'a',
    'time': 't',

    'error': 'e',  # ошибка
    'unknown_impact': 'h',  # неизвестное воздействие

    # 'model': 'y',  # модель
}

default_params = {
    'delimiter': '_',
    'default_idx': 0
}

OPERATORS = ('+', '-', '*', '/', '**')
BRAKETS = '()'

# class Parser:
#     def __init__(self):
#         var_names = variable_names
#         params = default_params
#
#     def parse_expr(self, expr: str):
#         pass


def parse_expr_old(expr: str):
    expr = expr.replace(' ', '')
    i = 0
    # FIXME: last_idx
    last_idx = 0
    res = ''
    x_names = {}  # ключи - индексы, значения - списки с именами
    u_names = {}  # ключи - индексы, значения - списки с именами
    tao_values = {}  # ключи - имена, значения - задержка  ???
    a_names = []  #

    while i < len(expr):
        s = expr[i]
        name = ''


        if s == variable_names['obj']:
            name, idx, tao, i = parse_var(expr, i, last_idx)
            # if idx in x_names:
            #     pass
            print('name =', name)
            print('idx =', idx)
            print('tao =', tao)
            print('i =', i)
        elif s == variable_names['control']:
            name, idx, tao, i = parse_var(expr, i, last_idx)
            print('name =', name)
            print('idx =', idx)
            print('tao =', tao)
            print('i =', i)
        elif s == variable_names['coefficient']:
            name, idx, _, i = parse_var(expr, i, last_idx)
            print('name =', name)
            print('idx =', idx)
            print('i =', i)
        elif s == variable_names['error']:
            pass
        elif s == variable_names['unknown_impact']:
            pass
        else:
            res += s
            i += 1
        # if name:
        res += name

    return res


def parse_var(expr: str, i: int, last_idx: int) -> Union[Exception, Tuple[str, int, int, int]]:
    var_control = variable_names['control']
    var_obj = variable_names['obj']
    var_coef = variable_names['coefficient']
    var_time = variable_names['time']
    default_idx = default_params['default_idx']
    delimiter = default_params['delimiter']

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
                    idx = last_idx + 1
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
                    name = s + str(idx) + '_' + str(idx)
                else:
                    name = s + str(default_idx) + '_' + str(tao)
                return name, idx, tao, i
            elif s == var_coef:
                message = 'Указание временной задержки у коэффициентов не поддерживаеся.'
                raise ValueError(message)
        else:
            message = get_error_message(reality=expr[i], position=i)
            raise ValueError(message)

    if i >= len(expr):
        if s == var_coef:
            if idx == -1:
                idx = last_idx + 1
            name = s + str(idx)
            return name, idx, 0, i
        elif s in (var_control, var_obj):
            message = 'Ожидается временная задержка'
            raise ValueError(message)


def parse_lag(expr: str, i: int, var_time: str) -> Union[Exception, Tuple[int, int]]:
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
        return tao, i + 1
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
    if sub_str:
        yield sub_str


def parse_expr(expr: str):
    expr = expr.replace(' ', '')
    # FIXME: last_idx
    last_idx = 0
    res = ''
    x_names = {}  # ключи - индексы, значения - списки с именами
    u_names = {}  # ключи - индексы, значения - списки с именами
    tao_values = {}  # ключи - имена, значения - задержка  ???
    a_names = []  #

    current_name = ''
    for token in get_token(expr):
        if token == variable_names['coefficient']:
            current_name = token
        elif token in (variable_names['obj'], variable_names['control']):
            pass
        elif token == '.':
            raise ValueError('Неожиданный символ "."')
        else:
            res += token


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


def main():
    # r = parse_expr('a*x(t-1)+a*u(t-1)+a*(x(t-2)+u(t-2))')
    # r = parse_expr('arctg(x(t-1))')
    # print(r)

    for t in get_token('sin(123)'):  # arctg(a_0*x(t-1)+3.9)
        print('->', t)


if __name__ == '__main__':
    main()
