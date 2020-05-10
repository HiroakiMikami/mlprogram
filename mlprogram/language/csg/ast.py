from mlprogram.ast.ast import Node, Leaf, Field


def Rectangle(w: int, h: int) -> Node:
    return Node("Rectangle", [
        Field("width", "number", Leaf("number", str(w))),
        Field("height", "number", Leaf("number", str(h)))
    ])


def Circle(r: int) -> Node:
    return Node("Circle", [
        Field("r", "number", Leaf("number", str(r)))
    ])


def Translation(x: int, y: int, child: Node) -> Node:
    return Node("Translation", [
        Field("x", "number", Leaf("number", str(x))),
        Field("y", "number", Leaf("number", str(y))),
        Field("child", "CSG", child)
    ])


def Rotation(theta_degree: int, child: Node) -> Node:
    return Node("Rotation", [
        Field("theta", "number", Leaf("number", str(theta_degree))),
        Field("child", "CSG", child)
    ])


def Union(a: Node, b: Node) -> Node:
    return Node("Union", [
        Field("a", "CSG", a),
        Field("b", "CSG", b)
    ])


def Difference(a: Node, b: Node) -> Node:
    return Node("Difference", [
        Field("a", "CSG", a),
        Field("b", "CSG", b)
    ])
