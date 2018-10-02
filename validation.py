def check_brackets(expr: str, brackets: str='()') -> int:
    opening, closing = brackets[::2], brackets[1::2]
    stack = []
    for c in expr:
        if c in opening:
            stack.append(opening.index(c))
        elif c in closing:
            if stack and stack[-1] == closing.index(c):
                stack.pop()
            else:
                return expr.index(c)
    return -1


def main():
    s = 'a0*x(t-1)+a_1*u(t-1)+a_2*(x(t-2)+u(t-2))'
    i = check_brackets(s)
    print(i)
    print(s[i])

if __name__ == '__main__':
    main()
