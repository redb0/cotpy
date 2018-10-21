def flatten(l):
    return [x for sublist in l for x in sublist]


def is_rect_matrix(m, sub_len=None) -> bool:
    l = len(m[0])
    for val in m:
        if len(val) != (sub_len if sub_len else l):
            return False
    return True
