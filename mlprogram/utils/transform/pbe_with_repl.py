from typing import List, Dict, Any, cast, Callable, Optional, Set
from mlprogram.utils import Token
from mlprogram.asts import AST, Node, Leaf
from mlprogram.interpreters import Interpreter, SequentialProgram, Reference


class ToEpisode:
    def __init__(self, to_ast: Optional[Callable[[Any], AST]] = None,
                 remove_used_reference: bool = False):
        self.to_ast = to_ast
        self.remove_used_reference = remove_used_reference

    def __call__(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        input = entry["input"]
        ground_truth = cast(SequentialProgram[Any], entry["ground_truth"])
        gt_refs = {statement.reference: statement.code
                   for statement in ground_truth.statements}

        def find_refs(ast: AST) -> List[Reference]:
            if isinstance(ast, Node):
                retval = []
                for field in ast.fields:
                    if isinstance(field.value, list):
                        for value in field.value:
                            retval.extend(find_refs(value))
                    else:
                        retval.extend(find_refs(field.value))
                return retval
            elif isinstance(ast, Leaf):
                if isinstance(ast.value, Reference):
                    return [ast.value]
            return []

        retval: List[Dict[str, Any]] = []
        refs: Set[Reference] = set()
        for i, statement in enumerate(ground_truth.statements):
            ref = statement.reference
            rs = list(refs)
            rs.sort(key=lambda r: r.name)
            xs = {key: value for key, value in entry.items()
                  if key not in set(["input", "ground_truth"])}
            xs["input"] = input
            xs["ground_truth"] = gt_refs[ref]
            xs["reference"] = [Token(None, r) for r in rs]
            xs["code"] = SequentialProgram(ground_truth.statements[:(i + 1)])
            retval.append(xs)
            refs.add(ref)
            if self.remove_used_reference:
                assert self.to_ast is not None
                used = find_refs(self.to_ast(gt_refs[ref]))
                for r in used:
                    if r in refs:
                        refs.remove(r)

        return retval


class EvaluateCode:
    def __init__(self, interpreter: Interpreter):
        self.interpreter = interpreter

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        code = entry["code"]
        reference = entry["reference"]
        refs = [token.value for token in reference]
        result = self.interpreter.eval_references(code)
        variables = [result[ref] for ref in refs]
        entry["variables"] = variables
        return entry
