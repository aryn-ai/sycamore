import hashlib
import os
from io import IOBase
from typing import Union
import threading

from diskcache import Cache


class CacheManager:
    def __init__(self, cache_loc: str):
        self._cache = Cache(directory=cache_loc)

    def get(self, hash_key: str):
        return self._cache.get(hash_key)

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)

    def get_hash_key(self, file_path: Union[str, IOBase]) -> str:
        hash_sha256 = hashlib.sha256()
        if isinstance(file_path, IOBase):
            hash_sha256.update(file_path.read())
        else:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(block)
        return hash_sha256.hexdigest()
