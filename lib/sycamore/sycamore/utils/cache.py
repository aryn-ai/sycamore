from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Optional

import boto3
import diskcache
from botocore.exceptions import ClientError

BLOCK_SIZE = 1048576  # 1 MiB


class Cache:
    def get(self, hash_key: str):
        pass

    def set(self, hash_key: str, hash_value):
        pass

    @staticmethod
    def get_hash_context(data: bytes) -> hashlib._Hash:
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(data)
        return hash_sha256

    @staticmethod
    def get_hash_context_file(file_path: str, hash_context: Optional[hashlib._Hash] = None) -> hashlib._Hash:
        if not hash_context:
            hash_context = hashlib.sha256()
        with open(file_path, "rb") as file:
            while True:
                file_buffer = file.read(BLOCK_SIZE)
                if not file_buffer:
                    break
                hash_context.update(file_buffer)
        return hash_context


class DiskCache(Cache):
    def __init__(self, cache_loc: str):
        self._cache = diskcache.Cache(directory=cache_loc)

    def get(self, hash_key: str):
        return self._cache.get(hash_key)

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)


def s3_cache_deserializer(kwargs):
    return S3Cache(**kwargs)


class S3Cache(Cache):
    def __init__(self, s3_path: str, freshness_in_seconds: int = -1):
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
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise

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
