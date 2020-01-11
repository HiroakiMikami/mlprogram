from typing import List, Callable, Union
from nl2prog.language.ast import AST
from nl2prog.language.action \
    import ActionSequence, ast_to_action_sequence


def to_action_sequence(code: str,
                       parse: Callable[[str], AST],
                       tokenize: Callable[[str], List[str]],
                       ) -> Union[ActionSequence, None]:
    ast = parse(code)
    if ast is None:
        return None
    return ast_to_action_sequence(ast, tokenizer=tokenize)
