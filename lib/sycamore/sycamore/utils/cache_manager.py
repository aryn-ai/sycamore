import hashlib
import os
from io import IOBase
from typing import Union
import threading

from diskcache import Cache


class CacheManager:
    _lock = threading.Lock()
    _cache = Cache(directory=os.path.join("/tmp/SycamoreCache", "PDFMinerCache"))

    @classmethod
    def get(cls, hash_key: str):
        return cls._cache.get(hash_key)

    @classmethod
    def set(cls, hash_key: str, hash_value):
        cls._cache.set(hash_key, hash_value)

    @classmethod
    def get_hash_key(cls, file_path: Union[str, IOBase]) -> str:
        hash_sha256 = hashlib.sha256()
        if isinstance(file_path, IOBase):
            hash_sha256.update(file_path.read())
        else:
            with cls._lock:
                with open(file_path, "rb") as f:
                    for block in iter(lambda: f.read(4096), b""):
                        hash_sha256.update(block)
        return hash_sha256.hexdigest()
