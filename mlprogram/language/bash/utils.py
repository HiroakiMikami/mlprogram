from typing import Union

from mlprogram.action.ast import Root


def is_subtype(subtype: Union[str, Root], basetype: Union[str, Root]) -> bool:
    if basetype == "Node" and subtype != "str":
        return True
    if basetype == subtype:
        return True
    return False
