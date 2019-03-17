"""
Модуль настроек библиотеки.

:Authors:
    - Vladimir Voronov
"""


# список зарезервированных имен переменных, используемых при разборе исходного выражения:
# coefficient - имя коэффициентов;
# output      - имя выходов модели;
# input       - имя входов модели (управляющее воздействие);
# add_input   - имя дополнительных входов.
expr_vars = {  # 'expr.'
    'coefficient': 'a',
    'output': 'x',
    'input': 'u',
    'add_input': 'z',
    # 'expr.error': 'e',  # ошибка
    # 'expr.unknown_impact': 'h',  # неизвестное воздействие
}

# список зарезервированных имен переменных, используемых при синтезе закона управления:
# predicted_output    - прогнозируемая переменная выходов;
# predicted_input     - прогнозируемая переменная входов (управляющее воздействие);
# predicted_add_input - прогнозируемая переменная дополнительных входов модели;
# trajectory          - желаемая траетория движения объекта.
control_law_vars = {  # 'control_law.'
    'predicted_output': 'xp',
    'predicted_input': 'up',
    'predicted_add_input': 'zp',
    'trajectory': 'xt',
}

# дополнительные параметры
parameters = {
    'time': 't',
    'delimiter': '_',
    'default_index': 0,
    'auto_index': False
}

# в mypy есть TypedDict
# можно так T = TypedDict('T', {'key1': str, 'key2': int})
