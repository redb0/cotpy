from random import randint

import pytest

from exprparse import expr_parser


def idfn(val):
    return f'expr={val}'

a = [
    ('a', 'a0', {}, {}, [('a0', 0)]),
    ('a1', 'a1', {}, {}, [('a1', 1)]),
    ('a25', 'a25', {}, {}, [('a25', 25)]),
    ('a_3', 'a3', {}, {}, [('a3', 3)]),
    ('a_46', 'a46', {}, {}, [('a46', 46)]),
]


def get_names():
    names = ('x', 'u')
    data = []

    for name in names:
        if name == 'u':
            item = (name + '(t-0)', name + '0_0', {}, {name + '0_0': (0, 0)}, [])
        elif name == 'x':
            item = (name + '(t-0)', name + '0_0', {name + '0_0': (0, 0)}, {}, [])
        data.append(item)

        for d in ['', '_']:
            i_1 = randint(0, 9)
            i_2 = randint(0, 9)
            i_3 = randint(10, 999)
            i_4 = randint(10, 999)
            if name == 'u':
                item = (
                    name + d + str(i_1) + '(t-' + str(i_2) + ')',
                    name + str(i_1) + '_' + str(i_2),
                    {},
                    {name + str(i_1) + '_' + str(i_2): (i_1, i_2)}, []
                )
                item_1 = (
                    name + d + str(i_3) + '(t-' + str(i_4) + ')',
                    name + str(i_3) + '_' + str(i_4),
                    {},
                    {name + str(i_3) + '_' + str(i_4): (i_3, i_4)}, []
                )
            elif name == 'x':
                item = (
                    name + d + str(i_1) + '(t-' + str(i_2) + ')',
                    name + str(i_1) + '_' + str(i_2),
                    {name + str(i_1) + '_' + str(i_2): (i_1, i_2)},
                    {}, []
                )
                item_1 = (
                    name + d + str(i_3) + '(t-' + str(i_4) + ')',
                    name + str(i_3) + '_' + str(i_4),
                    {name + str(i_3) + '_' + str(i_4): (i_3, i_4)},
                    {}, []
                )

            data.append(item)
            data.append(item_1)
    return data


@pytest.mark.parametrize('expect_a', a, ids=idfn)
def test_parse_expr_with_a(expect_a):
    real_a = expr_parser.parse_expr(expect_a[0])
    assert real_a == expect_a[1:]


@pytest.mark.parametrize('expect_xu', get_names(), ids=idfn)
def test_parse_expr(expect_xu):
    real_a = expr_parser.parse_expr(expect_xu[0])
    assert real_a == expect_xu[1:]
