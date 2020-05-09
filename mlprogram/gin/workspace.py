from typing import Dict, Any, Optional
from contextlib import contextmanager


class Workspace:
    objects: Dict[str, Any] = dict()


def put(key: str, value: Any) -> None:
    Workspace.objects[key] = value


def get(key: str) -> Optional[Any]:
    return Workspace.objects.get(key, None)


@contextmanager
def use_workspace():
    old_objects = Workspace.objects
    Workspace.objects = dict()
    yield
    Workspace.objects = old_objects
