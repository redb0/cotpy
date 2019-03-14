"""
Модуль настроек библиотеки.

Словарь variable_names содержит имена переменных, 
используемых в математических выражениях.
    Ключи:
        - obj               : Переменная, обозначающая выход объекта, по умолчанию - x.
        - control           : Переменная, обозначающая вход объекта 
                              или управляющее воздействие, по умолчанию - u.
        - coefficient       : Переменная, обозначающая неизвестные 
                              коэффициенты модели, по умолчанию - a.
        - time              : Переменная, обозначающая временные такты, по умолчанию - t.
        - predicted_outputs : Переменная, обозначающая прогнозное значение 
                              выхода оюъекта, по умолчанию - xp.
        - predicted_inputs  : Переменная, обозначающая прогнозное значение входа 
                              объекта/управляющего воздействия, по умолчанию - up.
        - trajectory        : Переменная, обозначающая желаемую 
                              траекторию движения объекта, по умолчанию - xt.
        - error             : Не используется.
        - unknown_impact    : Не используется.

Словарь default_params содержит прочие параметры.
    Ключи:
        - delimiter   : Разделитель, использующийся в именах переменных, по умолчанию - '_'.
        - default_idx : Индекс переменной, по умолчанию - 0.

:Authors:
    - Vladimir Voronov
"""

from typing import Dict, Union


expr_vars = {  # 'expr.'
    'coefficient': 'a',
    'output': 'x',
    'input': 'u',
    'add_input': 'z',
    # 'expr.error': 'e',  # ошибка
    # 'expr.unknown_impact': 'h',  # неизвестное воздействие
}

control_law_vars = {  # 'control_law.'
    'predicted_output': 'xp',
    'predicted_input': 'up',
    'predicted_add_input': 'zp',
    'trajectory': 'xt',
}

parameters = {
    'time': 't',
    'delimiter': '_',
    'default_index': 0,
    'auto_index': False
}

# в mypy есть TypedDict
# можно так T = TypedDict('T', {'key1': str, 'key2': int})
default_params: Dict[str, Union[str, int]] = {
    'delimiter': '_',
    'default_idx': 0,
}
