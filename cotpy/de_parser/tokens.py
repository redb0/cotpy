class TokenKind:
    """Виды токенов"""
    IDENTIFIER = 2
    SIGN = 3
    BRACKET = 4
    # NUMBER = 5
    INT = 6
    FLOAT = 7
    # DOT = 8

    tokens = []  # all tokens

for k, v in TokenKind.__dict__.items():
    if isinstance(v, int) and not k.startswith('__'):
        TokenKind.tokens.append(v)


class Token:
    def __init__(self, kind: int, s: str, start_idx: int, end_idx: int):
        """
        Создание токена.
        :param kind:      вид токена
        :type kind:       int
        :param s:         идентифицированная строка
        :type s:          str
        :param start_idx: индекс начала токена в исходной строке
        :type start_idx:  int
        :param end_idx:   индекс конца токена в исходной строке
        :type end_idx:    int
        """
        self._kind = kind
        self._s = s
        self._start_idx = start_idx
        self._end_idx = end_idx

    def to_tuple(self):
        return self._kind, self._s, self._start_idx, self._end_idx

    @property
    def kind(self) -> int:
        return self._kind

    def get_str(self) -> str:
        return self._s

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @property
    def end_idx(self) -> int:
        return self._end_idx

    def __repr__(self):
        return f'Token({self._kind}, {repr(self._s)}, {self._start_idx}, {self._end_idx})'
