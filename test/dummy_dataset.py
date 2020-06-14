from typing import List
from mlprogram.gin import workspace
from mlprogram.asts import AST, Node, Leaf, Field
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


def is_subtype(subtype: str, basetype: str) -> bool:
    if subtype == basetype:
        return True
    if subtype in set(["Name", "Number", "List"]) and basetype == "Value":
        return True
    return False


def string(value: str):
    return Leaf("string", value)


def number(value: int):
    return Leaf("number", str(value))


def Name(value: str):
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
traindata = [
    {"input": ["x is assigned the value of 0"],
     "ground_truth": [Assign("x", Number(0))]},
    {"input": ["dump the value of xy"],
     "ground_truth": [FunctionCall("print", [Name("xy")])]},
    {"input": ["dump the value of xy and x"],
     "ground_truth": [FunctionCall("print", [Name("xy"), Name("x")])]}
]
test_dataset = ListDataset([
    {"input": ["x is assigned the value of 4"],
     "ground_truth": [Assign("x", Number(4))]},
    {"input": ["dump the value of xy"],
     "ground_truth": [FunctionCall("print", [Name("xy")])]},
    {"input": ["dump the value of xy and x"],
     "ground_truth": [FunctionCall("print", [Name("xy"), Name("x")])]}
])


def prepare_dataset(dataset_path: str, num_repeat: int) -> None:
    dataset = []
    for _ in range(num_repeat):
        dataset.extend(traindata)
    workspace.put(dataset_path, {
        "train": ListDataset(dataset),
        "test": test_dataset,
        "valid": test_dataset
    })
