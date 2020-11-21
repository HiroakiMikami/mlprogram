from typing import Generic, List, TypeVar

Code = TypeVar("Code")
Error = TypeVar("Error")


class Analyzer(Generic[Code, Error]):
    def __call__(self, code: Code) -> List[Error]:
        raise NotImplementedError
