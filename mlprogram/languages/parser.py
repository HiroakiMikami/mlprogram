from typing import Generic, Optional, TypeVar

from mlprogram.languages import AST

Code = TypeVar("Code")


class Parser(Generic[Code]):
    def parse(self, code: Code) -> Optional[AST]:
        raise NotImplementedError

    def unparse(self, ast: AST) -> Optional[Code]:
        raise NotImplementedError
