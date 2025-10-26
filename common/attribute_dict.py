from typing import Any, Dict, List, Union

class AttrDict(dict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    @staticmethod
    def _convert(value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return AttrDict(value)
        if isinstance(value, list):
            return [AttrDict._convert(v) for v in value]
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._convert(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        other: Dict = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = v

    def __getattribute__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        if dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        if dict.__contains__(self, name):
            del self[name]
        else:
            object.__delattr__(self, name)

    def to_dict(self) -> dict:
        def conv(obj: Any) -> Any:
            if isinstance(obj, AttrDict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [conv(v) for v in obj]
            return obj
        return conv(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

