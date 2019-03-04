import warnings

from cotpy.de_parser.tokens import TokenKind
from cotpy.de_parser.lexer import tokenize
from cotpy.settings import expr_vars, parameters

# синтаксический анализатор

start_state = 1
final_error_state = 0


class TaoError(Exception):
    def __init__(self, message):
        self._msg = message

    def __str__(self):
        return self._msg


class VarIndexError(Exception):
    def __init__(self, message):
        self._msg = message

    def __str__(self):
        return self._msg


class VarNameError(Exception):
    def __init__(self, message):
        self._msg = message

    def __str__(self):
        return self._msg


class UnexpectedCharacter(Exception):
    def __init__(self, message):
        self._msg = message

    def __str__(self):
        return self._msg


class VarSyntaxError(SyntaxError):
    def __init__(self, message):
        self._msg = message

    def __str__(self):
        return self._msg


_var_fsm_states = {
    'name': 2,
    'c_name': 3,
    'delimiter': -1,
    'delimiter_tao': -3,
    'c_delimiter': -2,
    'index': 4,
    'c_index': 5,
    'tao': 6
}


def variable_fsm(ch: str, last_state: int) -> int:
    if last_state == start_state:
        if ch in (expr_vars['output'], expr_vars['input'],
                  expr_vars['add_input']):
            new_state = _var_fsm_states['name']
        elif ch == expr_vars['coefficient']:  # a
            new_state = _var_fsm_states['c_name']
        else:
            new_state = final_error_state
    else:
        if ch == parameters['delimiter']:
            if last_state == _var_fsm_states['name']:
                new_state = _var_fsm_states['delimiter']
            elif last_state == _var_fsm_states['c_name']:
                new_state = _var_fsm_states['c_delimiter']
            elif last_state == _var_fsm_states['index']:
                new_state = _var_fsm_states['delimiter_tao']
            else:
                new_state = final_error_state
        elif last_state in (_var_fsm_states['delimiter'],
                            _var_fsm_states['name'],
                            _var_fsm_states['index']) and ch.isdigit():  # for x, u, z
            new_state = _var_fsm_states['index']
        elif last_state in (_var_fsm_states['c_delimiter'],
                            _var_fsm_states['c_name'],
                            _var_fsm_states['c_index']) and ch.isdigit():  # for a
            new_state = _var_fsm_states['c_index']
        elif last_state in (_var_fsm_states['delimiter_tao'],
                            _var_fsm_states['tao']) and ch.isdigit():
            new_state = _var_fsm_states['tao']
        else:
            new_state = final_error_state
    return new_state


def var_parse(s: str):
    idx, tao = None, None
    state = 1
    last_state = 1
    idx_idx = 0
    i = 0
    while i < len(s):
        state = variable_fsm(s[i], state)
        if state != final_error_state:
            if state in (_var_fsm_states['index'],
                         _var_fsm_states['c_index']) and idx_idx == 0:
                idx_idx = i
            last_state = state
        else:
            if last_state in (_var_fsm_states['delimiter'],
                              _var_fsm_states['delimiter_tao']):
                raise VarIndexError(f'Требуется индекс ({i} символ)')
            elif last_state == _var_fsm_states['delimiter_tao']:
                raise TaoError(f'Требуется величина задержки ({i} символ)')
            elif last_state == _var_fsm_states['tao']:
                raise TaoError(f'Некорректная величина задержки ({i} символ)')
            elif last_state in (_var_fsm_states['name'], _var_fsm_states['c_name']):
                raise VarNameError(f'Некорректное имя переменной ({i} символ)')
            elif last_state in (_var_fsm_states['index'], _var_fsm_states['c_index']):
                raise VarIndexError(f'Некорректный индекс ({i} символ)')
        i += 1
    if i >= len(s):
        if last_state in (_var_fsm_states['delimiter'],
                          _var_fsm_states['c_delimiter']):
            raise VarIndexError(f'Требуется индекс ({i} символ)')
        elif last_state == _var_fsm_states['delimiter_tao']:
            raise TaoError(f'Требуется величина задержки ({i} символ)')
    if last_state == _var_fsm_states['index']:
        idx = int(s[idx_idx:])
    elif last_state == _var_fsm_states['c_index']:
        idx = int(s[idx_idx:])
    elif last_state == _var_fsm_states['tao']:
        idx = int(s[idx_idx:len(s)-len(s.split('_')[-1]) - 1])
        tao = int(s.split('_')[-1])

    return s[0], idx, tao


def compound_var_parse(start_idx: int, tokens):
    i = start_idx
    state = start_state
    var_info = None
    new_tao = None
    no_delay = False
    while state != 3 and i < len(tokens):
        token = tokens[i]
        s = token.get_str()
        if token.kind == TokenKind.IDENTIFIER:
            len_s = len(s)
            if s[0] in expr_vars.values() and (len_s == 1 or
                                               (len_s >= 1 and not s[1].isalpha())):
                var_info = var_parse(s)
                no_delay = var_info[2] is None
                if var_info[0] in (expr_vars['output'], expr_vars['input'],
                                   expr_vars['add_input']):
                    if no_delay:
                        state = 2
                    else:
                        state = 3
                else:
                    state = 3
            else:
                if s[0] == parameters['time']:
                    if len(s) > 1:
                        raise UnexpectedCharacter(f'Неизвестный символ {repr(s[1:])}, ожидается знак "-"')
                    if state == -1:
                        state = -2
                    else:
                        raise UnexpectedCharacter('Неизвестный символ, ожидается открывающая скобка')
                else:
                    raise UnexpectedCharacter(f'Неизвестный символ {repr(s)}, ожидается {repr(parameters["time"])}')
        elif token.kind == TokenKind.BRACKET and no_delay:
            if state == 2:
                state = -1
            elif state == -4:
                state = 3
            else:
                raise UnexpectedCharacter(f'Неожиданный символ {repr(s)}. '
                                          f'Ожидается величина задержки')
        elif token.kind == TokenKind.SIGN and no_delay:
            if state == -2 and s[0] == '-':
                state = -3
        elif token.kind == TokenKind.INT and no_delay:
            if state == -3:
                new_tao = int(s)
                state = -4
        else:
            if not no_delay:
                raise UnexpectedCharacter(f'Неожиданный символ {repr(s)}. '
                                          f'Переменная {repr(var_info[0])} имеет задержку = {var_info[2]}')
        i += 1

    if state == 2:
        raise TaoError(f'Требуется задержка для переменной {repr(var_info[0])} вида "(t-1)"')
    elif state == -1:
        raise VarSyntaxError('Ожидается задержка вида "(t-1)"')
    elif state == -2:
        raise VarSyntaxError('Ожидается знак "-"')
    elif state == -3:
        raise VarSyntaxError('Ожидается величина задержки')
    elif state == -4:
        raise VarSyntaxError('Ожидается закрывающая скобка')

    if new_tao is not None:
        return i, (var_info[0], var_info[1], new_tao)
    else:
        return i, var_info


def get_name(var_info) -> str:
    name = var_info[0] + str(var_info[1])
    if len(var_info) > 2 and var_info[2] is not None:
        name += '_' + str(var_info[2])
    return name


def get_auto_index(l, default_idx: int=0) -> int:
    if l:
        return max([i for i in l]) + 1
    else:
        return default_idx


def parse(tokens):
    i = 0
    inv_special_symbols = {v: k for k, v in expr_vars.items()}
    res = {k: dict() for k in expr_vars.keys()}
    expr = ''
    while i < len(tokens):
        if tokens[i].kind == TokenKind.IDENTIFIER:
            s = tokens[i].get_str()
            name = ''
            if s[0] in expr_vars.values() and (len(s) == 1 or
                                               (len(s) >= 1 and not s[1].isalpha())):
                i, var_info = compound_var_parse(i, tokens)
                i -= 1
                new_idx = var_info[1]
                if var_info[0] in (expr_vars['coefficient'], expr_vars['add_input']):
                    if new_idx is None:
                        if parameters['auto_index']:
                            new_idx = get_auto_index([j[1] for j in res[inv_special_symbols[var_info[0]]].values()],
                                                     default_idx=parameters['default_index'])
                            warnings.warn(f'Индексы переменных {repr(var_info[0])} будут расставлены автоматически!')
                        else:
                            new_idx = parameters['default_index']
                            warnings.warn(f'Индексы переменных {repr(var_info[0])} будут одинаковы и равны {new_idx}!')
                        var_info = (var_info[0], new_idx, var_info[2])
                    name = get_name(var_info)  # var_info[0] + str(var_info[1])
                else:
                    if new_idx is None:
                        new_idx = parameters['default_index']
                    else:
                        new_idx = var_info[1]
                    name = get_name((var_info[0], new_idx, var_info[2]))
                if name not in res[inv_special_symbols[var_info[0]]].keys():
                    res[inv_special_symbols[var_info[0]]][name] = (new_idx, var_info[2])
            if name:
                expr += name
            else:
                expr += tokens[i].get_str()
        else:
            expr += tokens[i].get_str()
        i += 1
    return expr, res  # coefficients, var_inputs_outputs


def expr_parse(expr: str):
    expr = expr.replace(' ', '')
    tokens = list(tokenize(expr))
    # print('Токены:', tokens)
    expr, res = parse(tokens)
    # print('Выражение:', expr)
    # print('Результат:', res)
    return expr, res


def main():
    # var_parse:
    s = 'a0+a1-a2'
    s1 = 'a12'  # ('a', 12, None)
    s2 = 'a_3'  # ('x', 3, None)
    s3 = 'x_0_1'  # ('x', 0, 1)
    s4 = 'x1_4'  # ('x', 1, 4)
    s5 = 'z9_1'  # ('z', 9, 1)
    s6 = 'x23_3451'  # ('x', 23, 3451)
    s7 = 'x56'  # ('x', 56, None)
    s8 = 'x_32'  # ('x', 32, None)
    s9 = 'x'  # ('x', None, None)
    s10 = 'a'  # ('a', None, None)
    s11 = 'u_3_d'  # TaoError
    s12 = 'z_1_2e'  # TaoError
    s13 = 'xx0_1'  # VarNameError
    s14 = 'u0e_2'  # VarIndexError
    s15 = 'a2e_1'  # VarIndexError
    s16 = 'a2_1'  # VarIndexError
    s17 = 'z_f_1'  # VarIndexError
    s18 = 'u_1_'
    s19 = 'u_'
    s20 = 'u_t'
    s21 = 'x1_'
    s22 = 'a1_'

    st = 'x_0_1'  # +
    st1 = 'x1_3'  # +
    st2 = 'x(t-1)'
    st3 = 'x0(t-1)'  # +
    st4 = 'x_0(t-1)'  # +

    st5 = 'u_1'
    st6 = 'x_0(t-1'
    st7 = 'x_0(t-)'
    st8 = 'x_0(t1)'
    st9 = 'x_0(e-1)'
    st10 = 'x_0t-1)'
    st11 = 'u_1+'

    st13 = 'a0+a1*x0(t-1)+a2*u_0_1'

    st14 = 'a+a+a'
    st15 = 'z_0(t-1)+z1(t-2)+z2(t-3)'
    st16 = 'a0+a1*x1(t-1)+a2*u_2_1+a3*x1(t-2)+a4*z1(t-1)'
    st17 = 'a_0+a_1*sin(a_2*x(t-1))-a3*asin(a4*u(t-2))+5..4'

    print(expr_parse(st16))


if __name__ == '__main__':
    main()
