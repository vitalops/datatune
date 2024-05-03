from datatune.storage import Storage
import s3fs

class S3Storage(Storage):
    def __init__(self, s3: s3fs.S3FileSystem, bucket: str):
        self.s3 = s3
        self.bucket = bucket
        super().__init__()
    
    def _set(self, key: str, value: bytes) -> None:
        with self.s3.open(f"{self.bucket}/{key}", "wb") as f:
            f.write(value)
    
    def _get(self, key: str) -> bytes:
        try:
            with self.s3.open(f"{self.bucket}/{key}", "rb") as f:
                return f.read()
        except FileNotFoundError as fe:
            raise KeyError(key) from fe
    
    def __len__(self) -> int:
        return len(self.s3.ls(self.bucket))
    
    def __contains__(self, key: str) -> bool:
        return self.s3.exists(f"{self.bucket}/{key}")
    
    def keys(self):
        return [key[len(self.bucket) + 1:] for key in self.s3.ls(self.bucket)]
    
    def __delitem__(self, key: str) -> None:
        self.s3.rm(f"{self.bucket}/{key}")
    
    def clear(self) -> None:
        for key in self.keys():
            self.s3.rm(key)
    
    def size(self, key: str) -> int:
        return self.s3.info(f"{self.bucket}/{key}")["Size"]
