from typing import Dict, Any


class StateDict:
    def __init__(self, state_dict: Dict[str, Any]):
        self._state_dict = state_dict

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return {k.replace(key + ".", "", 1): value
                for k, value in self._state_dict.items()
                if k.startswith(key)}

    def state_dict(self) -> Dict[str, Any]:
        return self._state_dict
