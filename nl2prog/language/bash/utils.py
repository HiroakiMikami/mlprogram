from typing import Union

from nl2prog.language.ast import Root


def is_subtype(subtype: Union[str, Root], basetype: Union[str, Root]) -> bool:
    if basetype == "Node" and subtype != "str":
        return True
    if basetype == subtype:
        return True
    return False
