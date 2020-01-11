def is_subtype(subtype: str, basetype: str) -> bool:
    if basetype == "Node" and subtype != "str":
        return True
    if basetype == subtype:
        return True
    return False
