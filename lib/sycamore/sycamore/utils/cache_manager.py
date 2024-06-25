import hashlib
import pickle
import threading
from io import IOBase
from typing import Union, Any


class CacheManager:
    _lock = threading.Lock()

    @classmethod
    def get_cached_result(cls, file_path: str):
        with cls._lock:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    @classmethod
    def cache_result(cls, file_content: Any, file_path: str):
        with cls._lock:
            with open(file_path, "wb") as f:
                pickle.dump(file_content, f)

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
