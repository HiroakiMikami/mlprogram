import bashlex
from typing import Optional, cast
import mlprogram.asts as A
from .bashlex_ast_to_ast import bashlex_ast_to_ast


def parse(script: str) -> Optional[A.AST]:
    try:
        script = script.replace('”', '"').replace('“', '"')
        return bashlex_ast_to_ast(script, bashlex.parse(script)[0])
    except Exception as e:  # noqa
        return None


def unparse(ast: A.AST) -> Optional[str]:
    try:
        if isinstance(ast, A.Node):
            # Node
            n = ast.type_name
            if n == "Operator":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "List":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return "".join([token for token in elems if token is not None])
            elif n == "Pipe":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "Pipeline":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return "".join([token for token in elems if token is not None])
            elif n == "Compound":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                body = "".join([token for token in elems if token is not None])
                elems = [unparse(p) for p in cast(list, ast.fields[1].value)]
                if None in set(elems):
                    return None
                redirects = \
                    "".join([token for token in elems if token is not None])
                return f"{body} {redirects}"
            elif n == "If":
                # TODO deal with newline
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return \
                    " ".join([token for token in elems if token is not None])
            elif n == "For":
                # TODO deal with newline
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return "".join([token for token in elems if token is not None])
            elif n == "While":
                # TODO deal with newline
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return \
                    " ".join([token for token in elems if token is not None])
            elif n == "Until":
                # TODO deal with newline
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return \
                    " ".join([token for token in elems if token is not None])
            elif n == "Command":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return \
                    " ".join([token for token in elems if token is not None])
            elif n == "Function":
                elems = [unparse(p) for p in cast(list, ast.fields[1].value)]
                if None in set(elems):
                    return None
                elems = [token for token in elems if token is not None]
                body = "".join([token for token in elems if token is not None])
                name = unparse(cast(A.AST, ast.fields[0].value))
                if name is None:
                    return None
                return f"function {name}()" + "{" + body + "}"
            elif n == "Literal":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "Word":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return "".join([token for token in elems if token is not None])
            elif n == "Assign":
                elems = [unparse(p) for p in cast(list, ast.fields[0].value)]
                if None in set(elems):
                    return None
                return "".join([token for token in elems if token is not None])
            elif n == "ReservedWord":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "Parameter":
                p = unparse(cast(A.AST, ast.fields[0].value))
                if p is None:
                    return None
                return "${" + p + "}"
            elif n == "Tilde":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "Redirect":
                t = unparse(cast(A.AST, ast.fields[0].value))
                if t is None:
                    return None

                if cast(A.AST,
                        ast.fields[1].value).get_type_name() != "None":
                    heredoc = unparse(cast(A.AST, ast.fields[1].value))
                else:
                    heredoc = ""
                if heredoc is None:
                    return None

                if cast(A.AST,
                        ast.fields[2].value).get_type_name() != "None":
                    input = unparse(cast(A.AST, ast.fields[2].value))
                else:
                    input = ""
                if input is None:
                    return None

                if cast(A.AST,
                        ast.fields[3].value).get_type_name() != "None":
                    output = unparse(cast(A.AST, ast.fields[3].value))
                else:
                    output = ""
                if output is None:
                    return None

                value = f"{input}{t}{output}"
                if heredoc != "":
                    value = f"{value}\n{heredoc}"
                return value
            elif n == "Heredoc":
                return unparse(cast(A.AST, ast.fields[0].value))
            elif n == "ProcessSubstitution":
                command = unparse(cast(A.AST, ast.fields[0].value))
                t = unparse(cast(A.AST, ast.fields[1].value))
                if command is None or t is None:
                    return None
                return f"{t}({command})"
            elif n == "CommandSubstitution":
                command = unparse(cast(A.AST, ast.fields[0].value))
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
                print(n)
                assert(False)
        elif isinstance(ast, A.Leaf):
            # Token
            return ast.value
    except:  # noqa
        pass
    return None
