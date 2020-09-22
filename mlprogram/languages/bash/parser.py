import bashlex
from typing import Optional, cast, Callable, List, Union
import mlprogram.asts as A
from mlprogram.languages.bash.bashlex_ast_to_ast import bashlex_ast_to_ast


class Parser(object):
    def __init__(self, tokenize: Callable[[str], List[str]]):
        super().__init__()
        self.tokenize = tokenize

    def parse(self, script: str) -> Optional[A.AST]:
        try:
            script = script.replace('”', '"').replace('“', '"')
            return bashlex_ast_to_ast(script, bashlex.parse(script)[0],
                                      self.tokenize)
        except Exception as e:  # noqa
            return None

    def unparse(self, ast: A.AST) -> Optional[str]:
        def value_to_str(ast: Union[A.AST, List[A.AST]]) -> Optional[str]:
            try:
                if isinstance(ast, A.Node):
                    # Node
                    n = ast.type_name
                    if n == "Operator":
                        return value_to_str(ast.fields[0].value)
                    elif n == "List":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return "".join(
                            [token for token in elems if token is not None])
                    elif n == "Pipe":
                        return value_to_str(ast.fields[0].value)
                    elif n == "Pipeline":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return "".join(
                            [token for token in elems if token is not None])
                    elif n == "Compound":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        body = "".join(
                            [token for token in elems if token is not None])
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[1].value)]
                        if None in set(elems):
                            return None
                        redirects = \
                            "".join(
                                [token for token in elems if token is not None]
                            )
                        return f"{body} {redirects}"
                    elif n == "If":
                        # TODO deal with newline
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return \
                            " ".join(
                                [token for token in elems if token is not None]
                            )
                    elif n == "For":
                        # TODO deal with newline
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return "".join(
                            [token for token in elems if token is not None])
                    elif n == "While":
                        # TODO deal with newline
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return \
                            " ".join(
                                [token for token in elems if token is not None]
                            )
                    elif n == "Until":
                        # TODO deal with newline
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return \
                            " ".join(
                                [token for token in elems if token is not None]
                            )
                    elif n == "Command":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return \
                            " ".join(
                                [token for token in elems if token is not None]
                            )
                    elif n == "Function":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[1].value)]
                        if None in set(elems):
                            return None
                        elems = [token for token in elems if token is not None]
                        body = "".join(
                            [token for token in elems if token is not None])
                        name = value_to_str(cast(A.AST, ast.fields[0].value))
                        if name is None:
                            return None
                        return f"function {name}()" + "{" + body + "}"
                    elif n == "Literal":
                        return value_to_str(ast.fields[0].value)
                    elif n == "Word":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return "".join(
                            [token for token in elems if token is not None])
                    elif n == "Assign":
                        elems = [value_to_str(p)
                                 for p in cast(list, ast.fields[0].value)]
                        if None in set(elems):
                            return None
                        return "".join(
                            [token for token in elems if token is not None])
                    elif n == "ReservedWord":
                        return value_to_str(ast.fields[0].value)
                    elif n == "Parameter":
                        p = value_to_str(ast.fields[0].value)
                        if p is None:
                            return None
                        return "${" + p + "}"
                    elif n == "Tilde":
                        return value_to_str(ast.fields[0].value)
                    elif n == "Redirect":
                        t = value_to_str(ast.fields[0].value)
                        if t is None:
                            return None

                        if cast(A.AST,
                                ast.fields[1].value).get_type_name() != "None":
                            heredoc = value_to_str(ast.fields[1].value)
                        else:
                            heredoc = ""
                        if heredoc is None:
                            return None

                        if cast(A.AST,
                                ast.fields[2].value).get_type_name() != "None":
                            input = value_to_str(ast.fields[2].value)
                        else:
                            input = ""
                        if input is None:
                            return None

                        if cast(A.AST,
                                ast.fields[3].value).get_type_name() != "None":
                            output = value_to_str(ast.fields[3].value)
                        else:
                            output = ""
                        if output is None:
                            return None

                        value = f"{input}{t}{output}"
                        if heredoc != "":
                            value = f"{value}\n{heredoc}"
                        return value
                    elif n == "Heredoc":
                        return value_to_str(ast.fields[0].value)
                    elif n == "ProcessSubstitution":
                        command = value_to_str(ast.fields[0].value)
                        t = value_to_str(ast.fields[1].value)
                        if command is None or t is None:
                            return None
                        return f"{t}({command})"
                    elif n == "CommandSubstitution":
                        command = value_to_str(ast.fields[0].value)
                        if command is None:
                            return None
                        try:
                            bashlex.parse(f"$({command})")
                            return f"$({command})"
                        except:  # noqa
                            return f"`{command}`"
                    elif n == "None":
                        return ""
                    else:
                        assert(False)
                elif isinstance(ast, A.Leaf):
                    # Token
                    return ast.value
                elif isinstance(ast, list):
                    return "".join(map(lambda x: str(value_to_str(x)), ast))
            except:  # noqa
                pass
            return None
        return value_to_str(ast)
