from torch import nn


class Add(nn.Module):
    def forward(self, *args, **kwargs):
        args = list(args) + [arg for arg in kwargs.values()]
        assert len(args) > 0
        value = args[0]
        for arg in args[1:]:
            value = value + arg
        return value


class Mul(nn.Module):
    def forward(self, *args, **kwargs):
        args = list(args) + [arg for arg in kwargs.values()]
        assert len(args) > 0
        value = args[0]
        for arg in args[1:]:
            value = value * arg
        return value


class Sub(nn.Module):
    def forward(self, lhs, rhs):
        return lhs - rhs


class Div(nn.Module):
    def forward(self, lhs, rhs):
        return lhs / rhs


class IntDiv(nn.Module):
    def forward(self, lhs, rhs):
        return lhs // rhs


class Neg(nn.Module):
    def forward(self, value):
        return -value
