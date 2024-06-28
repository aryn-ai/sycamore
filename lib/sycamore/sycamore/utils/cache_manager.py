import hashlib

from diskcache import Cache


class CacheManager:
    def __init__(self, cache_loc: str):
        self._cache = Cache(directory=cache_loc)

    def get(self, hash_key: str):
        return self._cache.get(hash_key)

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)

    @staticmethod
    def get_hash_key(data: bytes) -> str:
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(data)
        return hash_sha256.hexdigest()
