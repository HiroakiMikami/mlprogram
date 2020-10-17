class Add(object):
    def __call__(self, *args, **kwargs):
        args = list(args) + [arg for arg in kwargs.values()]
        assert len(args) > 0
        value = args[0]
        for arg in args[1:]:
            value = value + arg
        return value


class Mul(object):
    def __call__(self, *args, **kwargs):
        args = list(args) + [arg for arg in kwargs.values()]
        assert len(args) > 0
        value = args[0]
        for arg in args[1:]:
            value = value * arg
        return value


class Sub(object):
    def __call__(self, lhs, rhs):
        return lhs - rhs


class Div(object):
    def __call__(self, lhs, rhs):
        return lhs / rhs


class IntDiv(object):
    def __call__(self, lhs, rhs):
        return lhs // rhs


class Neg(object):
    def __call__(self, value):
        return -value
