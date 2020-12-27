from typing import Any, Callable, List, Optional, Union

import bashlex

from mlprogram.languages import AST, Field, Leaf, Node


def bashlex_ast_to_ast(script: str,
                       bashlex_ast: Union[Any, bashlex.ast.node],
                       split_value: Callable[[str], List[str]]) \
        -> AST:
    """
    Convert the bash command into AST
    """
    class Visitor(bashlex.ast.nodevisitor):
        def __init__(self):
            self.value: Optional[AST] = None

        def to_leaf_value(self, t: str, value: str) -> List[Leaf]:
            return [Leaf(t, token)
                    for token in split_value(value)]

        def visitoperator(self, n, op):
            # Node(type_name="Operator", fileds={"op": Leaf(op)})
            self.value = \
                Node("Operator", [Field("op", "str",
                                        self.to_leaf_value("str", op))])
            return False

        def visitlist(self, n, parts):
            # Node(type_name="List", fileds={"parts": [...]})
            parts = [bashlex_ast_to_ast(script, p, split_value) for p in parts]
            self.value = Node("List", [Field("parts", "Node", parts)])
            return False

        def visitpipe(self, n, pipe):
            # Node(type_name="Pipe", fileds={"pipe": Leaf(pipe)})
            self.value = \
                Node("Pipe", [Field("pipe", "str",
                                    self.to_leaf_value("str", pipe))])
            return False

        def visitpipeline(self, n, parts):
            # Node(type_name="Pipeline", fields={"parts": [...]})
            parts = [bashlex_ast_to_ast(script, p, split_value) for p in parts]
            self.value = Node("Pipeline",
                              [Field("parts", "Node", parts)])
            return False

        def visitcompound(self, n, list, redirects):
            # Node(type_name="Compound", fileds={"list":[**list],
            #                                    "redirects":[**redicects]})
            list = [bashlex_ast_to_ast(script, x, split_value) for x in list]
            redirects = [bashlex_ast_to_ast(
                script, x, split_value) for x in redirects]
            self.value = Node("Compound", [
                Field("list", "Node", list),
                Field("redirects", "Node", redirects)])
            return False

        def visitif(self, node, parts):
            # Node(type_name="If", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x, split_value) for x in parts]
            self.value = Node("If", [Field("parts", "Node", parts)])
            return False

        def visitfor(self, node, parts):
            # Node(type_name="For", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x, split_value) for x in parts]
            self.value = Node("For", [Field("parts", "Node", parts)])
            return False

        def visitwhile(self, node, parts):
            # Node(type_name="While", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x, split_value) for x in parts]
            self.value = Node("While", [Field("parts", "Node", parts)])
            return False

        def visituntil(self, node, parts):
            # Node(type_name="Until", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x, split_value) for x in parts]
            self.value = Node("Until", [Field("parts", "Node", parts)])
            return False

        def visitcommand(self, n, parts):
            # Node(type_name="Command", fields={"parts":[...]})
            parts = [bashlex_ast_to_ast(script, x, split_value) for x in parts]
            self.value = Node("Command", [Field("parts", "Node", parts)])
            return False

        def visitfunction(self, n, name, body, parts):
            # Node(type_name="Funtion", fields={"name":Leaf(name),
            #                                   "body":[...]}
            body = [bashlex_ast_to_ast(script, x, split_value) for x in body]
            self.value = Node("Function", [
                Field("name", "str", self.to_leaf_value("str", name)),
                Field("body", "Node", body)])
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
                        children.append(Node(
                            "Literal",
                            [Field(
                                "value", "str",
                                self.to_leaf_value("str", text))]))
                children.append(bashlex_ast_to_ast(script, part, split_value))
                prev = children[-1].type_name
                offset = end
            if offset != n.pos[1]:
                text = word[offset - n.pos[0]:]
                if prev == "CommandSubstitution" and text.startswith(")"):
                    # Workaround for bashlex bug
                    text = text[1:]
                if len(text) != 0:
                    children.append(Node(
                        "Literal",
                        [Field("value", "str",
                               self.to_leaf_value("str", text))]))
            return children

        def visitword(self, n, word):
            word = script[n.pos[0]:n.pos[1]]
            children = self.splitword(n, word)
            self.value = Node("Word", [Field("value", "Node", children)])
            return False

        def visitassignment(self, n, word):
            word = script[n.pos[0]:n.pos[1]]
            children = self.splitword(n, word)
            self.value = Node("Assign", [Field("value", "Node", children)])
            return False

        def visitreservedword(self, n, word):
            # Node(type_name="ReservedWord", fileds={"word": Leaf(word)})
            self.value = \
                Node("ReservedWord",
                     [Field("word", "str",
                            self.to_leaf_value("str", word))])
            return False

        def visitparameter(self, n, value):
            # Node(type_name="Parameter", fileds={"value": Leaf(value)})
            self.value = \
                Node("Parameter",
                     [Field("value", "str",
                            self.to_leaf_value("str", value))])
            return False

        def visittilde(self, n, value):
            # Node(type_name="Tilde", fileds={"value": Leaf(value)})
            self.value = \
                Node(
                    "Tilde", [Field("value", "str",
                                    self.to_leaf_value("str", value))])
            return False

        def visitredirect(self, n, input, type, output, heredoc):
            # Node(type_name="Redirect", fields={"type":Leaf(type),
            #                                    "heredoc": heredoc or NoneNode
            #                                    "input": input or NoneNode
            #                                    "output": output or NoneNode
            heredoc = bashlex_ast_to_ast(script, heredoc, split_value) \
                if heredoc is not None \
                else Node("None", [])
            input = bashlex_ast_to_ast(script, input, split_value) \
                if input is not None \
                else Node("None", [])
            output = bashlex_ast_to_ast(script, output, split_value) \
                if output is not None \
                else Node("None", [])
            self.value = Node("Redirect", [
                Field("type", "str", self.to_leaf_value("str", type)),
                Field("heredoc", "Node", heredoc),
                Field("input", "Node", input),
                Field("output", "Node", output)
            ])
            return False

        def visitheredoc(self, n, value):
            # Node(type_name="Heredoc", fileds={"value": Leaf(value)})
            self.value = Node(
                "Heredoc", [Field("value", "str",
                                  self.to_leaf_value("str", value))])
            return False

        def visitprocesssubstitution(self, n, command):
            # Node(type_name="ProcessSubstitution",
            #      fileds={"command": command, "type": Leaf(type)})
            t = script[n.pos[0]]
            self.value = Node("ProcessSubstitution",
                              [Field(
                                  "command", "Node",
                                  bashlex_ast_to_ast(script, command,
                                                     split_value)),
                               Field("type", "str",
                                     self.to_leaf_value("str", t))])
            return False

        def visitcommandsubstitution(self, n, command):
            # Node(type_name="CommandSubstitution",
            #      fileds={"command": command})
            self.value = Node("CommandSubstitution",
                              [Field(
                                  "command", "Node",
                                  bashlex_ast_to_ast(script, command,
                                                     split_value))])
            return False
    if isinstance(bashlex_ast, bashlex.ast.node):
        visitor = Visitor()
        visitor.visit(bashlex_ast)
        assert visitor.value is not None
        return visitor.value
    else:
        return Leaf("str", str(bashlex_ast))
