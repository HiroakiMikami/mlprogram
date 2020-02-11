import bashlex
from typing import Optional
import nl2prog.language.ast as A
from .bashlex_ast_to_ast import bashlex_ast_to_ast


def parse(script: str) -> Optional[A.AST]:
    try:
        script = script.replace('”', '"').replace('“', '"')
        return bashlex_ast_to_ast(script, bashlex.parse(script)[0])
    except:  # noqa
        return None


def unparse(ast: A.AST) -> Optional[str]:
    try:
        if isinstance(ast, A.Node):
            # Node
            n = ast.type_name
            if n == "Operator":
                return unparse(ast.fields[0].value)
            elif n == "List":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return "".join(elems)
            elif n == "Pipe":
                return unparse(ast.fields[0].value)
            elif n == "Pipeline":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return "".join(elems)
            elif n == "Compound":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                body = "".join(elems)
                elems = [unparse(p) for p in ast.fields[1].value]
                if None in set(elems):
                    return None
                redirects = "".join(elems)
                return "{} {}".format(body, redirects)
            elif n == "If":
                # TODO deal with newline
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return " ".join(elems)
            elif n == "For":
                # TODO deal with newline
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return "".join(elems)
            elif n == "While":
                # TODO deal with newline
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return " ".join(elems)
            elif n == "Until":
                # TODO deal with newline
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return " ".join(elems)
            elif n == "Command":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return " ".join(elems)
            elif n == "Function":
                elems = [unparse(p) for p in ast.fields[1].value]
                if None in set(elems):
                    return None
                body = "".join(elems)
                name = unparse(ast.fields[0].value)
                if name is None:
                    return None
                return "function {}() ".format(name, body)
            elif n == "Literal":
                return unparse(ast.fields[0].value)
            elif n == "Word":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return "".join(elems)
            elif n == "Assign":
                elems = [unparse(p) for p in ast.fields[0].value]
                if None in set(elems):
                    return None
                return "".join(elems)
            elif n == "ReservedWord":
                return unparse(ast.fields[0].value)
            elif n == "Parameter":
                p = unparse(ast.fields[0].value)
                if p is None:
                    return None
                return "${" + p + "}"
            elif n == "Tilde":
                return unparse(ast.fields[0].value)
            elif n == "Redirect":
                t = unparse(ast.fields[0].value)
                if t is None:
                    return None

                if ast.fields[1].value.type_name != "None":
                    heredoc = unparse(ast.fields[1].value)
                else:
                    heredoc = ""
                if heredoc is None:
                    return None

                if ast.fields[2].value.type_name != "None":
                    input = unparse(ast.fields[2].value)
                else:
                    input = ""
                if input is None:
                    return None

                if ast.fields[3].value.type_name != "None":
                    output = unparse(ast.fields[3].value)
                else:
                    output = ""
                if output is None:
                    return None

                value = "{}{}{}".format(input, t, output)
                if heredoc != "":
                    value = "{}\n{}".format(value, heredoc)
                return value
            elif n == "Heredoc":
                return unparse(ast.fields[0].value)
            elif n == "ProcessSubstitution":
                command = unparse(ast.fields[0].value)
                t = unparse(ast.fields[1].value)
                if command is None or t is None:
                    return None
                return "{}({})".format(t, command)
            elif n == "CommandSubstitution":
                command = unparse(ast.fields[0].value)
                if command is None:
                    return None
                try:
                    bashlex.parse("$({})".format(command))
                    return "$({})".format(command)
                except:  # noqa
                    return "`{}`".format(command)
            elif n == "None":
                return ""
            else:
                print(n)
                assert(False)
        else:
            # Token
            return ast.value
    except:  # noqa
        return None
