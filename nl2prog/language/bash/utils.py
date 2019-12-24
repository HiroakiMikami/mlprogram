from nl2prog.language.nl2code.action import NodeType


def is_subtype(subtype: NodeType, basetype: NodeType) -> bool:
    subtype = subtype.type_name
    basetype = basetype.type_name
    if basetype == "Node" and subtype != "str":
        return True
    if basetype == subtype:
        return True
    return False
