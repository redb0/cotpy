import warnings


def flatten(l):
    return [x for sublist in l for x in sublist]


def is_rect_matrix(m, sub_len=0, min_len=0) -> bool:
    l = len(m[0])
    for val in m:
        if min_len:
            if len(val) < min_len:
                return False
        else:
            if len(val) != (sub_len if sub_len else l):
                return False
    return True


def normalize_kwargs(kw, alias_map=None):
    res = dict()
    if alias_map is None:
        alias_map = dict()
    for canonical, alias in alias_map.items():
        values, seen_key = [], []
        if canonical not in alias:
            if canonical in kw:
                values.append(kw.pop(canonical))
                seen_key.append(canonical)
        for a in alias:
            if a in kw:
                values.append(kw.pop(a))
                seen_key.append(a)
        if values:
            res[canonical] = values[0]
            if len(values) > 1:
                warnings.warn(f'Все псевдонимы kwargs {seen_key!r} относятся к '
                              f'{canonical!r}. Будет применен только {seen_key[0]!r}')
    res.update(kw)
    return res


def bar(**kwargs):
    print(kwargs)
    m = {
        'aaa': ['a'],
        'bbb': ['b']
    }
    kw = normalize_kwargs(kwargs, alias_map=m)
    print(kw)


def main():
    bar(a=3, bbb=4, c=90)

if __name__ == '__main__':
    main()

