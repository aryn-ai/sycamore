import hashlib
import json
import time
from typing import Any

import diskcache
from botocore.exceptions import ClientError


class Cache:
    def get(self, hash_key: str):
        pass

    def set(self, hash_key: str, hash_value):
        pass


class DiskCache(Cache):
    def __init__(self, cache_loc: str):
        self._cache = diskcache.Cache(directory=cache_loc)

    def get(self, hash_key: str):
        return self._cache.get(hash_key)

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)


class S3Cache(Cache):
    def __init__(self, s3_client, s3_path: str, freshness_in_seconds: int = -1):
        self._s3_client = s3_client
        self._s3_path = s3_path
        self._freshness_in_seconds = freshness_in_seconds

    def _get_s3_bucket_and_key(self, key):
        parts = self._s3_path.replace("s3://", "").strip("/").split("/", 1)
        return parts[0], "/".join([parts[1], key]) if len(parts) == 2 else key

    def get(self, key: str):
        try:
            bucket, key = self._get_s3_bucket_and_key(key)
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            content = json.loads(response["Body"])

            # If enforcing freshness, we require cached data to have metadata
            if (
                self._freshness_in_seconds >= 0
                and self._freshness_in_seconds + content.get("cached_at", 0) < time.time()
            ):
                return None
            data = content["value"]
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise

    def set(self, key: str, value: Any):
        bucket, key = self._get_s3_bucket_and_key(key)

        content = {"value": value, "cached_at": time.time()}

        json_str = json.dumps(content, sort_keys=True, indent=2)
        self._s3_client.put_object(Body=json_str, Bucket=bucket, Key=key)


class CacheManager:
    def __init__(self, cache: Cache):
        self._cache = cache

    def get(self, hash_key: str):
        return self._cache.get(hash_key)

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)

    @staticmethod
    def get_hash_key(data: bytes) -> str:
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(data)
        return hash_sha256.hexdigest()
