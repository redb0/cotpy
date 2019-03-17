from cotpy.de_parser.tokens import TokenKind, Token

# final_successful_state = 3
final_error_state = 0
start_state = 1


def init_fsm(n: int):  # finite-state machine
    """
    Инициализация конечных автоматов. 
    Начальное состояние каждого автомата задается переменной start_state. 
    Значение которой по умолчанию равно 1.
    :param n: количество автоматов.
    :type n : int
    :return: список (длиной n) состояний автоматов
    """
    states = [start_state for _ in range(n)]
    return states


def identifier_fsm(c: str, last_state: int) -> int:
    """
    Конечный автомат распознования идентификатора.
    :param c         : текущий символ строки
    :type c          : str
    :param last_state: предыдущее состояние автомата
    :type last_state : int
    :return: токен идентификатора, в противном случае состояние ошибки (final_error_state)
    """
    if (last_state == start_state and c.isalpha()) or (last_state == TokenKind.IDENTIFIER and
                                                       (c.isalnum() or c == '_')):
        return TokenKind.IDENTIFIER
    else:
        return final_error_state  # ошибочное завершение


def math_sings_fsm(c: str, last_state: int) -> int:
    # last_state, *_ = last_state
    if ((last_state == start_state) and (c in '+-/*')) or ((last_state == TokenKind.SIGN) and (c == '*')):
        return TokenKind.SIGN
    # elif (last_state == start_state) and c == '*':
    #     return 4
    else:
        return final_error_state


def brackets_fsm(c: str, last_state: int) -> int:
    # last_state, *_ = last_state
    if last_state == start_state and c in '()':
        return TokenKind.BRACKET
    else:
        return final_error_state


# def int_fsm(c: str, last_state: int) -> int:
#     if (last_state == start_state and c.isdigit()) or (last_state == tokens['int']):
#         return tokens['int']
#     else:
#         return final_error_state
#
#
# def float_fsm(c: str, last_state: int) -> int:
#     if last_state == start_state and c.isdigit():
#         return tokens['float']
#     else:
#         return final_error_state


def numbers_fsm(c: str, last_state: int) -> int:
    # last_state, dot_flag = last_state
    if (last_state in (start_state, TokenKind.INT)) and c.isdigit():  # TokenKind.INT
        new_state = TokenKind.INT
    elif last_state == TokenKind.INT and c == '.':
        new_state = TokenKind.FLOAT
    elif last_state == start_state and c == '.':
        new_state = -1  # TokenKind.DOT  # промежуточное состояние
    elif (last_state in (-1, TokenKind.FLOAT)) and c.isdigit():  # TokenKind.DOT
        new_state = TokenKind.FLOAT
    else:
        # if last_state in (TokenKind.INT, TokenKind.FLOAT):
        #     last_state = TokenKind.NUMBER
        new_state = final_error_state
    return new_state  # , last_state


def identify(c: str, states):
    """
    Идентификация символа.
    Распознаются 4 вида лексем: идентификаторы (имена переменных, функций), 
    знаки основных математических операций, скобки, числа.
    :param c     : символ, требующий идентификации
    :type c      : str
    :param states: список состояний каждого автомата. 
                   Состояние каждого автомата представлено целым числом.
    :type states : list
    :return: обновленный список состояний автоматов
    """
    states[0] = identifier_fsm(c, states[0])  # идентификаторы
    states[1] = math_sings_fsm(c, states[1])  # матем. знаки
    states[2] = brackets_fsm(c, states[2])  # скобки
    states[3] = numbers_fsm(c, states[3])
    return states


def get_final_state(states):
    for state in states:
        if state in TokenKind.tokens:
            return state
    return 0


def all_error_state(states):
    return all([True if state == final_error_state else False for state in states])


def all_start_state(states):
    return all([True if state == start_state else False for state in states])


def tokenize(s: str):
    number_fsm = 4  # len(tokens) - 2
    last_idx_final_state = 0
    start_idx = 0
    last_final_state = 0
    states = init_fsm(number_fsm)
    i = 0
    while i < len(s):
        states = identify(s[i], states)
        if all_error_state(states):
            if last_final_state == 0:
                token = Token(last_final_state, s[start_idx:start_idx + 1], start_idx, start_idx)
                i = start_idx
                start_idx += 1
            else:
                token = Token(last_final_state, s[start_idx:i], start_idx, i - 1)
                i = last_idx_final_state
                start_idx = last_idx_final_state + 1
            states = init_fsm(number_fsm)
            yield token
        else:
            last_final_state = get_final_state(states)
            last_idx_final_state = i
        i += 1
    if ((last_final_state != 0) or not all_start_state(states)) and (i >= len(s)):
        token = Token(last_final_state, s[start_idx:i], start_idx, i - 1)
        yield token


def main():
    s = 'a0+a_2**a3'
    s4 = 'x0(t-1)'
    s6 = '1.34'
    s7 = '.04'
    s8 = '1.'
    s9 = '23.5.2'
    s10 = '.'
    s11 = '1+2.3-39'
    s12 = '1*4 -7'
    s13 = 'sin(u0(t-2))'
    s14 = '|'
    s15 = '1'
    s16 = 'x'
    s17 = '..0'
    s18 = '1..'
    print(list(tokenize(s.replace(' ', ''))))

if __name__ == '__main__':
    main()
