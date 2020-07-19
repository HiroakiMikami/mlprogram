from typing import List, Dict, Any, cast, Callable, Optional, Set
from mlprogram.utils import Reference
from mlprogram.asts import AST, Node, Leaf


class ToEpisode:
    def __init__(self, to_ast: Optional[Callable[[Any], AST]] = None,
                 remove_used_reference: bool = False):
        self.to_ast = to_ast
        self.remove_used_reference = remove_used_reference

    def __call__(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        input = entry["input"]
        references = cast(List[Reference], entry["references"])
        ground_truth = cast(Dict[Reference, Any], entry["ground_truth"])
        variables = cast(Dict[Reference, Any], entry["variables"])

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
        for ref in references:
            rs = list(refs)
            retval.append({
                "input": input,
                "ground_truth": ground_truth[ref],
                "references": rs,
                "variables": [variables[r] for r in rs]
            })
            refs.add(ref)
            if self.remove_used_reference:
                assert self.to_ast is not None
                used = find_refs(self.to_ast(ground_truth[ref]))
                for r in used:
                    if r in refs:
                        refs.remove(r)

        return retval
