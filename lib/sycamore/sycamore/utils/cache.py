from __future__ import annotations
import hashlib
import json
import logging
import threading
from datetime import timedelta
from pathlib import Path
import time
from tempfile import SpooledTemporaryFile
from typing import Any, Optional, Union, BinaryIO

import diskcache
from botocore.exceptions import ClientError

BLOCK_SIZE = 1048576  # 1 MiB
DDB_CACHE_TTL: int = int(timedelta(days=10).total_seconds())


class HashContext:
    """
    This is a wrapper class for the hash context as Python/mypy/IDE does not like accessing _Hash from hashlib
    """

    def __init__(self, /, algorithm: str = "sha256", copy_from=None) -> None:
        if copy_from:
            self.hash_obj = copy_from.hash_obj.copy()
        else:
            self.hash_obj = hashlib.new(algorithm, usedforsecurity=False)

    def copy(self) -> HashContext:
        return HashContext(copy_from=self)

    def update(self, data: bytes) -> None:
        self.hash_obj.update(data)

    def hexdigest(self) -> str:
        return self.hash_obj.hexdigest()


class Cache:

    def __init__(self):
        self.mutex = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, hash_key: str):
        pass

    def set(self, hash_key: str, hash_value):
        pass

    def inc_hits(self):
        with self.mutex:
            self.cache_hits += 1

    def inc_misses(self):
        with self.mutex:
            self.cache_misses += 1

    def get_hit_rate(self):
        with self.mutex:
            total = self.cache_hits + self.cache_misses
            if total == 0:
                return 0.0
            return self.cache_hits / total

    @staticmethod
    def get_hash_context(data: bytes, hash_ctx: Optional[HashContext] = None) -> HashContext:
        if not hash_ctx:
            hash_ctx = HashContext()
        hash_ctx.update(data)
        return hash_ctx

    @staticmethod
    def get_hash_context_file(
        file_path: Union[str, BinaryIO, SpooledTemporaryFile], hash_ctx: Optional[HashContext] = None
    ) -> HashContext:
        if not hash_ctx:
            hash_ctx = HashContext()

        if isinstance(file_path, BinaryIO) or isinstance(file_path, SpooledTemporaryFile):
            return Cache._update_ctx(file_path, hash_ctx)
        else:
            with open(file_path, "rb") as file:
                return Cache._update_ctx(file, hash_ctx)

    @staticmethod
    def _update_ctx(file_obj: Union[BinaryIO, SpooledTemporaryFile], hash_ctx: HashContext):
        while True:
            file_buffer = file_obj.read(BLOCK_SIZE)
            if not file_buffer:
                break
            hash_ctx.update(file_buffer)
        return hash_ctx

    @staticmethod
    def copy_and_hash_file(src, dest, hash_ctx: Optional[HashContext] = None) -> HashContext:
        if not hash_ctx:
            hash_ctx = HashContext()
        to_close = []
        if isinstance(src, (str, Path)):
            src = open(src, "rb")
            to_close.append(src)
        else:
            src.seek(0)
        if isinstance(dest, (str, Path)):
            dest = open(dest, "wb")
            to_close.append(dest)

        chunk = 262144  # 256kB
        with memoryview(bytearray(chunk)) as buf:
            while (got := src.readinto(buf)) > 0:
                if got >= chunk:
                    hash_ctx.update(buf)
                    dest.write(buf)
                else:
                    with buf[:got] as part:
                        hash_ctx.update(part)
                        dest.write(part)

        for f in to_close:
            f.close()
        return hash_ctx


class DiskCache(Cache):
    def __init__(self, cache_loc: str):
        super().__init__()
        self._cache = diskcache.Cache(directory=cache_loc)

    def get(self, hash_key: str):
        v = self._cache.get(hash_key)
        if v is not None:
            self.inc_hits()
            return v
        self.inc_misses()
        return v

    def set(self, hash_key: str, hash_value):
        self._cache.set(hash_key, hash_value)


def s3_cache_deserializer(kwargs):
    return S3Cache(**kwargs)


class S3Cache(Cache):
    def __init__(self, s3_path: str, freshness_in_seconds: int = -1):
        from mypy_boto3_s3.client import S3Client

        super().__init__()
        self._s3_path = s3_path
        self._freshness_in_seconds = freshness_in_seconds
        self._s3_client: Optional[S3Client] = None

    def _get_s3_bucket_and_key(self, key):
        parts = self._s3_path.replace("s3://", "").strip("/").split("/", 1)
        return parts[0], "/".join([parts[1], key]) if len(parts) == 2 else key

    def get(self, key: str):
        if not self._s3_client:
            import boto3

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
                self.inc_misses()
                return None
            data = content["value"]
            self.inc_hits()
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise

    def set(self, key: str, value: Any):
        if not self._s3_client:
            import boto3

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


class DynamoDBCache(Cache):
    """
    A DynamoDB cache items are specified as follows:

    ddb://<region_name>/<table_name>[/<hash_key_name>]

    where 'hash_key_name' defaults to 'hash_key' if left unspecified.

    This cache uses an attribute called 'expire_at' to use DynamoDB's TTL feature for cache expiration.
    The DynamoDB table backing this cache must have TTL enabled on this attribute for this to work properly.
    """

    def __init__(self, path: str, ttl: int = DDB_CACHE_TTL):
        import boto3

        super().__init__()
        scheme, _, region_name, table_name, hash_key_name = self.parse_path(path)
        self.hash_key_name = hash_key_name if hash_key_name is not None else "hash_key"
        self.client = boto3.client("dynamodb", region_name=region_name)
        self.table_name = table_name
        self.ttl = ttl

    @staticmethod
    def parse_path(path: str):
        assert path.startswith("ddb://"), "DynamoDB cache paths must start with ddb://"

        parts = path.split("/", 5)
        if len(parts) < 4:
            raise ValueError("DynamoDB cache paths must have 'region_name' (us-east-1, e.g.) and 'table_name'")

        if len(parts) == 4:
            return (*parts, None)

        return tuple(parts)

    def get(self, hash_key: str) -> Optional[bytes]:
        key = {self.hash_key_name: hash_key}
        res: dict[Any, Any] = {}
        try:
            res = self.client.get_item(TableName=self.table_name, Key=key)
        except ClientError as error:
            logging.error(f"Error calling get_item({key}) on {self.table_name} : {error}")

        if res is not None:
            if item := res.get("Item"):
                if payload := item.get("payload"):
                    self.inc_hits()
                    return payload.value
        self.inc_misses()
        return None

    def set(self, hash_key: str, hash_value: bytes):
        ttl = int(time.time()) + self.ttl
        item = {
            self.hash_key_name: {"S": hash_key},
            "payload": {"B": hash_value},
            "expire_at": {"N": f"{ttl}"},
        }
        self.client.put_item(TableName=self.table_name, Item=item)


def cache_from_path(path: Optional[str]) -> Optional[Cache]:
    if path is None:
        return None
    if path.startswith("s3://"):
        return S3Cache(path)
    if path.startswith("ddb://"):
        return DynamoDBCache(path)
    if path.startswith("/"):
        return DiskCache(path)
    if Path(path).is_dir():
        return DiskCache(path)

    raise ValueError(
        f"Unable to interpret {path} as path for cache. Expected s3://, /... or a directory path that exists"
    )
