from typing import Any, List, Tuple, Callable, Optional
from bisect import bisect_left


class TopKElement:
    """
    Hold top-k elements
    """

    def __init__(self, k: int,
                 handle_deleted_element: Optional[Callable[[Any],
                                                           None]] = None):
        """
        Parameters
        ----------
        k: int
            The maximum number of elements
        handle_deleted_element: Optional[Callable[[Any], None]]
            The function called when the element is deleted
        """
        self._elems = []
        self._k = k
        self._handle_deleted_element = handle_deleted_element

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
            if self._handle_deleted_element is not None:
                self._handle_deleted_element(elem)
            return
        self._elems.insert(index, (-score, elem))
        if self._handle_deleted_element is not None:
            for _, elem in self._elems[self._k:]:
                self._handle_deleted_element(elem)

        self._elems = self._elems[:self._k]
