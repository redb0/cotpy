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


def parse_var():
    pass
