from typing import Generic
from typing import TypeVar
from typing import List

Code = TypeVar("Code")


class Expander(Generic[Code]):
    def expand(self, code: Code) -> List[Code]:
        raise NotImplementedError

    def unexpand(self, code: List[Code]) -> Code:
        raise NotImplementedError
