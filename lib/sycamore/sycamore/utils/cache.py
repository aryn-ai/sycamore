from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Optional, Union

import boto3
import diskcache
from botocore.exceptions import ClientError

BLOCK_SIZE = 1048576  # 1 MiB


class HashWrapper:
    """
    This is a wrapper class for the hash context as Python/mypy/IDE does not like accessing _Hash from hashlib
    """

    def __init__(self, algorithm="sha256"):
        self.hash_obj = hashlib.new(algorithm)

    def update(self, data: list[bytes]):
        for d in data:
            self.hash_obj.update(d)

    def hexdigest(self):
        return self.hash_obj.hexdigest()


class Cache:

    def __init__(self):
        self.cache_hits = 0
        self.total_accesses = 0

    def get(self, hash_key: str):
        pass

    def set(self, hash_key: str, hash_value):
        pass

    def get_hit_rate(self):
        if self.total_accesses == 0:
            return 0.0
        return self.cache_hits / self.total_accesses

    @staticmethod
    def get_hash_context(data: Union[bytes, list[bytes]]) -> HashWrapper:
        if isinstance(data, bytes):
            data = [data]
        hash_ctx = HashWrapper()
        hash_ctx.update(data)
        return hash_ctx

    @staticmethod
    def get_hash_context_file(file_path: str, hash_context: Optional[HashWrapper] = None) -> HashWrapper:
        if not hash_context:
            hash_context = HashWrapper()
        with open(file_path, "rb") as file:
            while True:
                file_buffer = file.read(BLOCK_SIZE)
                if not file_buffer:
                    break
                hash_context.update([file_buffer])
        return hash_context


class DiskCache(Cache):
    def __init__(self, cache_loc: str):
        super().__init__()
        self._cache = diskcache.Cache(directory=cache_loc)

    def get(self, hash_key: str):
        v = self._cache.get(hash_key)
        if v is not None:
            self.cache_hits += 1
        self.total_accesses += 1
        return v

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)


def s3_cache_deserializer(kwargs):
    return S3Cache(**kwargs)


class S3Cache(Cache):
    def __init__(self, s3_path: str, freshness_in_seconds: int = -1):
        super().__init__()
        self._s3_path = s3_path
        self._freshness_in_seconds = freshness_in_seconds
        self._s3_client = None

    def _get_s3_bucket_and_key(self, key):
        parts = self._s3_path.replace("s3://", "").strip("/").split("/", 1)
        return parts[0], "/".join([parts[1], key]) if len(parts) == 2 else key

    def get(self, key: str):
        if not self._s3_client:
            self._s3_client = boto3.client("s3")
        try:
            assert self._s3_client is not None
            bucket, key = self._get_s3_bucket_and_key(key)
            response = self._s3_client.get_object(Bucket=bucket, Key=key)

            content = json.loads(response["Body"].read())

            # If enforcing freshness, we require cached data to have metadata
            if (
                self._freshness_in_seconds >= 0
                and self._freshness_in_seconds + content.get("cached_at", 0) < time.time()
            ):
                return None
            data = content["value"]
            self.cache_hits += 1
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise
        finally:
            self.total_accesses += 1

    def set(self, key: str, value: Any):
        if not self._s3_client:
            self._s3_client = boto3.client("s3")
        assert self._s3_client is not None
        bucket, key = self._get_s3_bucket_and_key(key)

        content = {"value": value, "cached_at": time.time()}

        json_str = json.dumps(content, sort_keys=True, indent=2)
        self._s3_client.put_object(Body=json_str, Bucket=bucket, Key=key)

    # The actual s3 client is not pickleable, This just says to pickle the wrapper, which can be used to
    # recreate the client on the other end.
    def __reduce__(self):

        kwargs = {"s3_path": self._s3_path, "freshness_in_seconds": self._freshness_in_seconds}

        return s3_cache_deserializer, (kwargs,)
