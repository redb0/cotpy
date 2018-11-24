from typing import Dict, Union

variable_names: Dict[str, str] = {
    'obj': 'x',
    'control': 'u',
    'coefficient': 'a',
    'time': 't',

    'predicted_outputs': 'xp',
    'predicted_inputs': 'up',

    'error': 'e',  # ошибка
    'unknown_impact': 'h',  # неизвестное воздействие
}

# в mypy есть TypedDict
# можно так T = TypedDict('T', {'key1': str, 'key2': int})
default_params: Dict[str, Union[str, int]] = {
    'delimiter': '_',
    'default_idx': 0,
}
