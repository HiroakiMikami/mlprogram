from typing import List
from typing import Union
from typing import Optional
from mlprogram.languages import AST, Node, Leaf, Field, Root
from mlprogram.languages import Token
from mlprogram.languages import Parser as BaseParser
from mlprogram.utils.data import ListDataset


# Definition of dummy language
"""
Program := Assign | FunctionCall
Assign := Name = Value
FunctionCall := Name(Value*)
Name := <token(string)>
Number := <token(integer)>
Name and Number are subtypes of Value
"""


def is_subtype(subtype: Union[str, Root], basetype: Union[str, Root]) -> bool:
    if isinstance(basetype, Root):
        return True
    if subtype == basetype:
        return True
    if subtype in set(["Name", "Number", "List"]) and basetype == "Value":
        return True
    return False


def tokenize(str: str) -> List[Token]:
    return list(map(lambda x: Token(None, x, x), str.split(" ")))


class Parser(BaseParser[AST]):
    def parse(self, x: AST) -> Optional[AST]:
        return x

    def unparse(self, x: AST) -> Optional[AST]:
        return x


def string(value: str):
    return Leaf("string", value)


def number(value: int):
    return Leaf("number", str(value))


def Name(value: Union[str, List[str]]):
    if isinstance(value, list):
        return Node("Name", [Field("value", "string",
                                   list(map(string, value)))])
    else:
        return Node("Name", [Field("value", "string", string(value))])


def Number(value: int):
    return Node("Number", [Field("value", "number", number(value))])


def Assign(var: str, value: AST):
    return Node("Assign", [Field("var", "Name", Name(var)),
                           Field("value", "Value", value)])


def FunctionCall(name: str, args: List[AST]):
    return Node("FunctionCall", [Field("name", "Name", Name(name)),
                                 Field("args", "Value", args)])


# Dataset
train_dataset = ListDataset([
    {"input": "x is assigned the value of 0",
     "ground_truth": Assign("x", Number(0))},
    {"input": "dump the value of xy",
     "ground_truth": FunctionCall("print", [Name(["x", "y"])])},
    {"input": "dump the value of xy and x",
     "ground_truth": FunctionCall("print", [Name(["x", "y"]), Name("x")])}
])
test_dataset = ListDataset([
    {"input": "x is assigned the value of 4",
     "ground_truth": Assign("x", Number(4))},
    {"input": "dump the value of xy",
     "ground_truth": FunctionCall("print", [Name(["x", "y"])])},
    {"input": "dump the value of xy and x",
     "ground_truth": FunctionCall("print", [Name(["x", "y"]), Name("x")])}
])
