from typing import Union

from mlprogram.action.ast import Root


def is_subtype(subtype: Union[str, Root], basetype: Union[str, Root]) -> bool:
    if isinstance(subtype, Root) or isinstance(basetype, Root):
        return subtype == basetype
    elif basetype == "CSG":
        return subtype != "number"
    else:
        return subtype == basetype
