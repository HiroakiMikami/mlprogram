import ast
import transpyle


def parse(code: str) -> ast.AST:
    """
    Return the AST of the code

    Parameters
    ----------
    code: str
        The code to be parsed

    Returns
    -------
    ast.AST
        The AST of the code
    """
    return ast.parse(code).body[0]


def unparse(ast: ast.AST) -> str:
    """
    Return the string of the AST

    Parameters
    ----------
    ast: ast.AST
        The AST to be unparsed

    Returns
    -------
    str
        The resulted string
    """
    unparser = transpyle.python.unparser.NativePythonUnparser()

    return unparser.unparse(ast)
