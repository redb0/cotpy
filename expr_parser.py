from typing import Tuple, Union


# TODO: пока только односимвольные имена
default_params = {
    'var_obj': 'x',
    'var_control': 'u',
    'var_coefficient': 'a',
    'var_time': 't',
    'var_error': 'e',  # ошибка
    'var_unknown_impact': 'h',  # неизвестное воздействие

    # ??????
    'model': 'y',  # модель
    'delimiter': '_',
    'default_idx': 0
}

OPERATORS = ('+', '-', '*', '/', '**')

# TODO: а может применить ООП? сделать класс Parser и у него будет поле со списком параметров?


def parse_expr(expr: str, params=default_params):
    expr = expr.replace(' ', '')
    i = 0
    # FIXME: last_idx
    last_idx = 0
    res = ''
    while i < len(expr):
        s = expr[i]
        name = ''
        if s == params['var_obj']:
            name, idx, tao, i = parse_var(expr, i, last_idx, params=default_params)
            print('name =', name)
            print('idx =', idx)
            print('tao =', tao)
            print('i =', i)
        elif s == params['var_control']:
            name, idx, tao, i = parse_var(expr, i, last_idx, params=default_params)
            print('name =', name)
            print('idx =', idx)
            print('tao =', tao)
            print('i =', i)
        elif s == params['var_coefficient']:
            name, idx, _, i = parse_var(expr, i, last_idx, params=default_params)
            print('name =', name)
            print('idx =', idx)
            print('i =', i)
        elif s == params['var_error']:
            pass
        elif s == params['var_unknown_impact']:
            pass
        else:
            res += s
            i += 1
        # if name:
        res += name

    return res


def parse_var(expr: str, i: int, last_idx: int,
              params=default_params) -> Union[Exception, Tuple[str, int, int, int]]:
    s = expr[i]
    idx = -1
    i += 1
    while i < len(expr) and expr[i] != ')':
        if expr[i] == params['delimiter']:
            i += 1
        elif expr[i].isdigit():
            idx, i = int_parse(expr, i)
        elif expr[i] in OPERATORS:
            if s == params['var_coefficient']:
                if idx == -1:
                    idx = last_idx + 1
                name = s + str(idx)  # params['delimiter']
                return name, idx, 0, i
            elif s in (params['var_control'], params['var_obj']):
                message = 'Некорректное указание временной задержки. Ожидается: "(". Текущий символ: ' + expr[i] + '.'
                raise ValueError(message)
        elif expr[i] == '(':
            i += 1
            if s in (params['var_control'], params['var_obj']):
                tao, i = parse_lag(expr, i, params['var_time'])
                if idx != -1:
                    name = s + str(idx) + '_' + str(idx)
                else:
                    name = s + str(params['default_idx']) + '_' + str(tao)
                return name, idx, tao, i
            elif s == params['var_coefficient']:
                message = 'Указание временной задержки у коэффициентов не поддерживаеся.'
                raise ValueError(message)
        else:
            message = get_error_message(reality=expr[i], position=i)
            raise ValueError(message)

    if i >= len(expr):
        if s == params['var_coefficient']:
            if idx == -1:
                idx = last_idx + 1
            name = s + str(idx)  # params['delimiter']
            return name, idx, 0, i
        elif s in (params['var_control'], params['var_obj']):
            message = 'Ожидается временная задержка'
            raise ValueError(message)

    # if expr[i] == ')':
    #     print('тут закрывающая скобка')


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
    r = parse_expr('a0*x(t-1)+a_1*u(t-1)+a_2*(x(t-2)+u(t-2))')
    print(r)

if __name__ == '__main__':
    main()
