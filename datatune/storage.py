from typing import Any, Union, List
from typing import Iterator

class Storage:

    def __getitem__(self, key: str) -> bytes:
        return self._get(key)
    
    def __setitem__(self, key: Union[str, List[str]], value: bytes) -> None:
        if isinstance(key, list):
            return self._pset(key, value)
        self._set(key, value)
    
    def __delitem__(self, key: str) -> None:
        raise NotImplementedError()
    
    def __contains__(self, key: str) -> bool:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def keys(self) -> Iterator[str]:
        raise NotImplementedError()
    
    def size(self, key: str) -> int:
        raise NotImplementedError()
    
    def clear(self) -> None:
        raise NotImplementedError()

    def get(self, key: str, default: bytes = b"") -> bytes:
        try:
            return self._get(key)
        except KeyError:
            return default

    def _set(self, key: str, value: bytes) -> None:
        raise NotImplementedError()

    def _pset(self, keys: List[str], value: List[bytes]) -> None:
        for key in keys:
            self._set(key, value)

    def _get(self, key: str) -> bytes:
        raise NotImplementedError()

    def _pget(self, keys: List[str]) -> List[bytes]:
        return [self._get(key) for key in keys]

