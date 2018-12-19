def check_brackets(expr: str, brackets: str='()') -> bool:
    opening, closing = brackets[::2], brackets[1::2]
    stack = []
    for c in expr:
        if c in opening:
            stack.append(opening.index(c))
        elif c in closing:
            if stack and stack[-1] == closing.index(c):
                stack.pop()
            else:
                # return expr.index(c)
                return False
    return not stack
