import bashlex
from typing import Union, Any
import nl2code.language.ast as A
from nl2code.language.action import NodeType


def bashlex_ast_to_ast(script: str,
                       bashlex_ast: Union[Any, bashlex.ast.node]) \
        -> A.AST:
    """
    Convert the bash command into AST
    """
    class Visitor(bashlex.ast.nodevisitor):
        def __init__(self):
            self.value = []

        def visitoperator(self, n, op):
            # Node(type_name="Operator", fileds={"op": Leaf(op)})
            self.value = \
                A.Node("Operator", [A.Field("op", "str", A.Leaf("str", op))])
            return False

        def visitlist(self, n, parts):
            # Node(type_name="List", fileds={"parts": [...]})
            parts = [bashlex_ast_to_ast(script, p) for p in parts]
            self.value = A.Node("List", [A.Field("parts", "Node", parts)])
            return False

        def visitpipe(self, n, pipe):
            # Node(type_name="Pipe", fileds={"pipe": Leaf(pipe)})
            self.value = \
                A.Node("Pipe", [A.Field("pipe", "str", A.Leaf("str", pipe))])
            return False

        def visitpipeline(self, n, parts):
            # Node(type_name="Pipeline", fields={"parts": [...]})
            parts = [bashlex_ast_to_ast(script, p) for p in parts]
            self.value = A.Node("Pipeline",
                                [A.Field("parts", "Node", parts)])
            return False

        def visitcompound(self, n, list, redirects):
            # Node(type_name="Compound", fileds={"list":[**list],
            #                                    "redirects":[**redicects]})
            list = [bashlex_ast_to_ast(script, x) for x in list]
            redirects = [bashlex_ast_to_ast(script, x) for x in redirects]
            self.value = A.Node("Compound", [
                A.Field("list", "Node", list),
                A.Field("redirects", "Node", redirects)])
            return False

        def visitif(self, node, parts):
            # Node(type_name="If", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x) for x in parts]
            self.value = A.Node("If", [A.Field("parts", "Node", parts)])
            return False

        def visitfor(self, node, parts):
            # Node(type_name="For", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x) for x in parts]
            self.value = A.Node("For", [A.Field("parts", "Node", parts)])
            return False

        def visitwhile(self, node, parts):
            # Node(type_name="While", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x) for x in parts]
            self.value = A.Node("While", [A.Field("parts", "Node", parts)])
            return False

        def visituntil(self, node, parts):
            # Node(type_name="Until", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x) for x in parts]
            self.value = A.Node("Until", [A.Field("parts", "Node", parts)])
            return False

        def visitcommand(self, n, parts):
            # Node(type_name="Command", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x) for x in parts]
            self.value = A.Node("Command", [A.Field("parts", "Node", parts)])
            return False

        def visitfunction(self, n, name, body, parts):
            # Node(type_name="Funtion", fields={"name":Leaf(name),
            #                                   "body":[...]}
            body = [bashlex_ast_to_ast(script, x) for x in body]
            self.value = A.Node("Function", [
                A.Field("name", "str", A.Leaf("str", name)),
                A.Field("body", "Node", body)])
            return False

        def splitword(self, n, word):
            offset = n.pos[0]
            parts = []
            children = []
            prev = None
            if hasattr(n, "parts"):
                parts = n.parts
            for part in parts:
                begin = part.pos[0]
                end = part.pos[1]
                if offset != begin:
                    text = word[offset - n.pos[0]:begin - n.pos[0]]
                    if prev == "CommandSubstitution" and text.startswith(")"):
                        # Workaround for bashlex bug
                        text = text[1:]
                    if len(text) != 0:
                        children.append(A.Node("Literal",
                                               [A.Field("value", "str",
                                                        A.Leaf("str", text))]))
                children.append(bashlex_ast_to_ast(script, part))
                prev = children[-1].type_name
                offset = end
            if offset != n.pos[1]:
                text = word[offset - n.pos[0]:]
                if prev == "CommandSubstitution" and text.startswith(")"):
                    # Workaround for bashlex bug
                    text = text[1:]
                if len(text) != 0:
                    children.append(A.Node("Literal",
                                           [A.Field("value", "str",
                                                    A.Leaf("str", text))]))
            return children

        def visitword(self, n, word):
            word = script[n.pos[0]:n.pos[1]]
            children = self.splitword(n, word)
            self.value = A.Node("Word", [A.Field("value", "Node", children)])
            return False

        def visitassignment(self, n, word):
            word = script[n.pos[0]:n.pos[1]]
            children = self.splitword(n, word)
            self.value = A.Node("Assign", [A.Field("value", "Node", children)])
            return False

        def visitreservedword(self, n, word):
            # Node(type_name="ReservedWord", fileds={"word": Leaf(word)})
            self.value = \
                A.Node("ReservedWord",
                       [A.Field("word", "str", A.Leaf("str", word))])
            return False

        def visitparameter(self, n, value):
            # Node(type_name="Parameter", fileds={"value": Leaf(value)})
            self.value = \
                A.Node("Parameter",
                       [A.Field("value", "str", A.Leaf("str", value))])
            return False

        def visittilde(self, n, value):
            # Node(type_name="Tilde", fileds={"value": Leaf(value)})
            self.value = \
                A.Node(
                    "Tilde", [A.Field("value", "str", A.Leaf("str", value))])
            return False

        def visitredirect(self, n, input, type, output, heredoc):
            # Node(type_name="Redirect", fields={"type":Leaf(type),
            #                                    "heredoc": heredoc or NoneNode
            #                                    "input": input or NoneNode
            #                                    "output": output or NoneNode
            heredoc = bashlex_ast_to_ast(script, heredoc) \
                if heredoc is not None \
                else A.Node("None", [])
            input = bashlex_ast_to_ast(script, input) if input is not None \
                else A.Node("None", [])
            output = bashlex_ast_to_ast(script, output) if output is not None \
                else A.Node("None", [])
            self.value = A.Node("Redirect", [
                A.Field("type", "str", A.Leaf("str", type)),
                A.Field("heredoc", "Node", heredoc),
                A.Field("input", "Node", input),
                A.Field("output", "Node", output)
            ])
            return False

        def visitheredoc(self, n, value):
            # Node(type_name="Heredoc", fileds={"value": Leaf(value)})
            self.value = A.Node(
                "Heredoc", [A.Field("value", "str", A.Leaf("str", value))])
            return False

        def visitprocesssubstitution(self, n, command):
            # Node(type_name="ProcessSubstitution",
            #      fileds={"command": command, "type": Leaf(type)})
            t = script[n.pos[0]]
            self.value = A.Node("ProcessSubstitution",
                                [A.Field("command", "Node",
                                         bashlex_ast_to_ast(script, command)),
                                 A.Field("type", "str", A.Leaf("str", t))])
            return False

        def visitcommandsubstitution(self, n, command):
            # Node(type_name="CommandSubstitution",
            #      fileds={"command": command})
            self.value = A.Node("CommandSubstitution",
                                [A.Field("command", "Node",
                                         bashlex_ast_to_ast(script, command))])
            return False
    if isinstance(bashlex_ast, bashlex.ast.node):
        visitor = Visitor()
        visitor.visit(bashlex_ast)
        return visitor.value
    else:
        return A.Leaf("str", str(bashlex_ast))


def parse(script: str) -> A.AST:
    script = script.replace('”', '"').replace('“', '"')
    return bashlex_ast_to_ast(script, bashlex.parse(script)[0])


def unparse(ast: A.AST) -> str:
    if isinstance(ast, A.Node):
        # Node
        n = ast.type_name
        if n == "Operator":
            return unparse(ast.fields[0].value)
        elif n == "List":
            return "".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Pipe":
            return unparse(ast.fields[0].value)
        elif n == "Pipeline":
            return "".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Compound":
            body = "".join([unparse(p) for p in ast.fields[0].value])
            redirects = "".join([unparse(p) for p in ast.fields[1].value])
            return "{} {}".format(body, redirects)
        elif n == "If":
            # TODO deal with newline
            return " ".join([unparse(p) for p in ast.fields[0].value])
        elif n == "For":
            # TODO deal with newline
            return "".join([unparse(p) for p in ast.fields[0].value])
        elif n == "While":
            # TODO deal with newline
            return " ".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Until":
            # TODO deal with newline
            return " ".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Command":
            return " ".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Function":
            body = "".join([unparse(p) for p in ast.fields[1].value])
            return "function {}() ".format(unparse(ast.fields[0].value), body)
        elif n == "Literal":
            return unparse(ast.fields[0].value)
        elif n == "Word":
            return "".join([unparse(p) for p in ast.fields[0].value])
        elif n == "Assign":
            return "".join([unparse(p) for p in ast.fields[0].value])
        elif n == "ReservedWord":
            return unparse(ast.fields[0].value)
        elif n == "Parameter":
            return "${" + unparse(ast.fields[0].value) + "}"
        elif n == "Tilde":
            return unparse(ast.fields[0].value)
        elif n == "Redirect":
            t = unparse(ast.fields[0].value)
            if ast.fields[1].value.type_name != "None":
                heredoc = unparse(ast.fields[1].value)
            else:
                heredoc = ""
            if ast.fields[2].value.type_name != "None":
                input = unparse(ast.fields[2].value)
            else:
                input = ""
            if ast.fields[3].value.type_name != "None":
                output = unparse(ast.fields[3].value)
            else:
                output = ""
            value = "{}{}{}".format(input, t, output)
            if heredoc != "":
                value = "{}\n{}".format(value, heredoc)
            return value
        elif n == "Heredoc":
            return unparse(ast.fields[0].value)
        elif n == "ProcessSubstitution":
            command = unparse(ast.fields[0].value)
            t = unparse(ast.fields[1].value)
            return "{}({})".format(t, command)
        elif n == "CommandSubstitution":
            command = unparse(ast.fields[0].value)
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


def is_subtype(subtype: NodeType, basetype: NodeType) -> bool:
    subtype = subtype.type_name
    basetype = basetype.type_name
    if basetype == "Node" and subtype != "str":
        return True
    if basetype == subtype:
        return True
    print(subtype, basetype)
    return False
