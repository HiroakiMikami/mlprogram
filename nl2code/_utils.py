from typing import Any, List, Tuple
from bisect import bisect_left


class TopKElement:
    """
    Hold top-k elements
    """
    def __init__(self, k: int):
        """
        Parameters
        ----------
        k: int
            The maximum number of elements
        """
        self._elems = []
        self._k = k

    @property
    def elements(self) -> List[Tuple[float, Any]]:
        """
        Returns the list of element
        """
        return list(map(lambda x: (-x[0], x[1]), self._elems))

    def add(self, score: float, elem: Any):
        """
        Add an element to the container

        Parameters
        ----------
        score: float
        elem: Any
        """
        index = bisect_left(list(map(lambda x: x[0], self._elems)), -score)
        if index > self._k:
            return
        self._elems.insert(index, (-score, elem))
        self._elems = self._elems[:self._k]
