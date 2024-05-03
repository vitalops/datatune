from datatune.storage import Storage


class MemoryStorage(Storage):
    def __init__(self):
        self.objects = {}
        super().__init__()

    def _set(self, key: str, value: bytes) -> None:
        self.objects[key] = value

    def _get(self, key: str) -> bytes:
        return self.objects[key]

    def __len__(self) -> int:
        return len(self.objects)

    def __contains__(self, key: str) -> bool:
        return key in self.objects
    
    def keys(self):
        return self.objects.keys()

    def __delitem__(self, key: str) -> None:
        del self.objects[key]

    def clear(self) -> None:
        self.objects.clear()
    
    def size(self, key: str) -> int:
        return len(self.objects[key])
