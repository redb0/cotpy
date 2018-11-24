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


# def normalize_kwargs(kw, alias_mapping=None, required=(), forbidden=(),
#                      allowed=None):
#     """Helper function to normalize kwarg inputs
#
#     The order they are resolved are:
#      1. aliasing
#      2. required
#      3. forbidden
#      4. allowed
#
#     This order means that only the canonical names need appear in
#     `allowed`, `forbidden`, `required`
#
#     Parameters
#     ----------
#     alias_mapping, dict, optional
#         A mapping between a canonical name to a list of
#         aliases, in order of precedence from lowest to highest.
#
#         If the canonical value is not in the list it is assumed to have
#         the highest priority.
#
#     required : iterable, optional
#         A tuple of fields that must be in kwargs.
#
#     forbidden : iterable, optional
#         A list of keys which may not be in kwargs
#
#     allowed : tuple, optional
#         A tuple of allowed fields.  If this not None, then raise if
#         `kw` contains any keys not in the union of `required`
#         and `allowed`.  To allow only the required fields pass in
#         ``()`` for `allowed`
#
#     Raises
#     ------
#     TypeError
#         To match what python raises if invalid args/kwargs are passed to
#         a callable.
#
#     """
#     # deal with default value of alias_mapping
#     if alias_mapping is None:
#         alias_mapping = dict()
#     # make a local so we can pop
#     kw = dict(kw)
#     # output dictionary
#     ret = dict()
#     # hit all alias mappings
#     for canonical, alias_list in six.iteritems(alias_mapping):
#
#         # the alias lists are ordered from lowest to highest priority
#         # so we know to use the last value in this list
#         tmp = []
#         seen = []
#         for a in alias_list:
#             try:
#                 tmp.append(kw.pop(a))
#                 seen.append(a)
#             except KeyError:
#                 pass
#         # if canonical is not in the alias_list assume highest priority
#         if canonical not in alias_list:
#             try:
#                 tmp.append(kw.pop(canonical))
#                 seen.append(canonical)
#             except KeyError:
#                 pass
#         # if we found anything in this set of aliases put it in the return
#         # dict
#         if tmp:
#             ret[canonical] = tmp[-1]
#             if len(tmp) > 1:
#                 warnings.warn("Saw kwargs {seen!r} which are all aliases for "
#                               "{canon!r}.  Kept value from {used!r}".format(
#                                   seen=seen, canon=canonical, used=seen[-1]))
#
#     # at this point we know that all keys which are aliased are removed, update
#     # the return dictionary from the cleaned local copy of the input
#     ret.update(kw)
#
#     fail_keys = [k for k in required if k not in ret]
#     if fail_keys:
#         raise TypeError("The required keys {keys!r} "
#                         "are not in kwargs".format(keys=fail_keys))
#     fail_keys = [k for k in forbidden if k in ret]
#     if fail_keys:
#         raise TypeError("The forbidden keys {keys!r} "
#                         "are in kwargs".format(keys=fail_keys))
#     if allowed is not None:
#         allowed_set = set(required) | set(allowed)
#         fail_keys = [k for k in ret if k not in allowed_set]
#         if fail_keys:
#             raise TypeError("kwargs contains {keys!r} which are not in "
#                             "the required {req!r} or "
#                             "allowed {allow!r} keys".format(
#                                 keys=fail_keys, req=required,
#                                 allow=allowed))
#     return ret


def normalize_kwargs(kw, alias_map=None):
    # по убыванию приоритета
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

