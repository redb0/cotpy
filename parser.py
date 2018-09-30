from typing import Tuple, Union

default_params = {
    "var_obj": "x",
    "var_control": "u",
    "var_coefficient": "a",
    "var_time": "t",
    "var_error": "e",  # ошибка
    "var_unknown_impact": "h",  # неизвестное воздействие

    # ??????
    "model": "y",  # модель
    "delimiter": "_"
}

OPERATORS = ('+', '-', '*', '/', '**')

# TODO: а может применить ООП? сделать класс Parser и у него будет поле со списком параметров?


def parse_expr(expr: str, params=default_params):
    expr = expr.replace(' ', '')
    i = 0
    while i < len(expr):
        s = expr[i]
        if s == params["var_obj"]:
            pass
        elif s == params["var_control"]:
            pass
        elif s == params["var_coefficient"]:
            pass
        elif s == params["var_error"]:
            pass
        elif s == params["var_unknown_impact"]:
            pass


def parse_var(expr: str, i: int, last_idx: int, delimiter: str='_',
              params=default_params) -> Union[Exception, Tuple[str, int, int, int]]:
    s = expr[i]
    idx = -1
    i += 1
    while i < len(expr) and expr[i] != ')':
        if expr[i] == delimiter:
            i += 1
        elif expr[i].isdigit():
            idx, i = int_parse(expr, i)
        elif expr[i] in OPERATORS:
            if s == params['var_coefficient']:
                if idx == -1:
                    idx = last_idx + 1
                name = s + delimiter + str(idx)
                return name, idx, 0, i
            elif s in (params['var_control'], params['var_obj']):
                message = 'Некорректное указание временной задержки. Ожидается: "(". Текущий символ: ' + expr[i] + '.'
                raise ValueError(message)
        elif expr[i] == '(':
            i += 1
            if s in (params['var_control'], params['var_obj']):
                tao, i = parse_lag(expr, i, params['var_time'])
                if idx != -1:
                    name = s + delimiter + str(idx)
                else:
                    name = s + delimiter + str(tao)
                return name, idx, tao, i
            elif s == params['var_coefficient']:
                message = 'Указание временной задержки у коэффициентов не поддерживаеся.'
                raise ValueError(message)

    if i >= len(expr):
        if s == params['var_coefficient']:
            if idx == -1:
                idx = last_idx + 1
            name = s + delimiter + str(idx)
            return name, idx, 0, i
        elif s in (params['var_control'], params['var_obj']):
            message = 'Ожидается временная задержка'
            raise ValueError(message)

    if expr[i] == ')':
        pass


def parse_lag(expr: str, i: int, var_time: str) -> Union[Exception, Tuple[int, int]]:
    tao = 0
    start = i
    while i <= len(expr) and expr[i] != ')':
        if i - start == 0:
            if expr[i] == var_time:
                i += 1
            else:
                message = 'Ожидается "' + var_time + '". Текущий символ "' + expr[i] + '"'
                raise ValueError(message)
        elif i - start == 1:
            if expr[i] == '-':
                i += 1
            else:
                message = 'Ожидается "' + var_time + '". Текущий символ "' + expr[i] + '"'
                raise ValueError(message)
        elif i - start == 2:
            if expr[i].isdigit():
                tao, i = int_parse(expr, i)
                break
            else:
                message = 'Ожидается величина задержки. Текущий символ "' + expr[i] + '"'
                raise ValueError(message)

    if expr[i] == ')':
        return tao, i + 1
    message = 'Ожидается ")". Текущий символ "' + expr[i] + '"'
    raise ValueError(message)


def int_parse(expr: str, i: int) -> Tuple[int, int]:
    numb = ''
    while i < len(expr) and expr[i].isdigit():
        numb += expr[i]
        i += 1
    return int(numb), i


def main():
    s = 't-1)'
    s1 = 't-25)'

    s2 = 't-36.0)'
    s3 = '-t1)'
    s4 = 't-a1)'
    s5 = 't-2s)'
    s6 = '7t-)'

    # tao, i = parse_lag(s6, 0, 't')
    # print(tao, i)

    v = 'x(t-1)'  # +
    v1 = 'u(t-1)'  # +
    v2 = 'x_0(t-1)'  # +
    v3 = 'u_0(t-1)'  # +
    v4 = 'a'  # +
    v5 = 'a1'  # +
    v6 = 'a_2'  # +

    v7 = 'a(t-1)'  # +
    v8 = 'a(t'  # +

    v9 = 'x'  # +
    v10 = 'u'  # +
    v11 = 'x_0'  # +
    v12 = 'u_0'  # +

    v13 = 'x2(t-1)'  # +
    v14 = 'u3(t-1)'  # +
    v15 = 'x1(t-5)'  # +
    v16 = 'u2(t-3)'  # +

    # v9 = '3a'

    name, idx, tao, i = parse_var(v4, 0, 0)
    print(name, idx, tao, i)
    # print(v[i])

if __name__ == '__main__':
    main()

