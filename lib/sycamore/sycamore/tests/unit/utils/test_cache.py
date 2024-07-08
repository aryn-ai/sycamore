import io
import json
from pathlib import Path
from unittest.mock import patch

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from sycamore.utils.cache import DiskCache, S3Cache
import hashlib


def get_hash(obj):
    json_str = json.dumps(obj, sort_keys=True)
    hash_obj = hashlib.sha256(json_str.encode("utf-8"))
    return hash_obj.hexdigest()


class TestDiskCache:
    def test_disk_cache(self, tmp_path: Path):
        cm = DiskCache(str(tmp_path))
        data1 = {"key": 1, "value": "one"}
        data2 = {"key": 1, "value": "one"}
        # insert-delete
        assert cm.get(get_hash(data1)) is None
        cm.set(get_hash(data1), data1)
        assert cm.get(get_hash(data1)) == data1

        # multiple records
        cm.set(get_hash(data2), data2)
        assert cm.get(get_hash(data1)) == data1
        assert cm.get(get_hash(data2)) == data2

        # update record
        cm.set(get_hash(data1), data2)
        assert cm.get(get_hash(data1)) == data2


class TestS3Cache:
    @patch("time.time", return_value=1000)
    def test_get_with_fresh_data(self, mock_time):
        s3_client = boto3.client("s3")
        stubber = Stubber(s3_client)
        s3_path = "s3://mybucket/myprefix"
        cache = S3Cache(s3_path)
        cache._s3_client = s3_client

        key = "testkey"
        value = "testvalue"

        raw_response = json.dumps({"value": value, "cached_at": 900})
        encoded_message = raw_response.encode()
        raw_stream = StreamingBody(io.BytesIO(encoded_message), len(encoded_message))
        response = {"Body": raw_stream}

        stubber.add_response("get_object", response, {"Bucket": "mybucket", "Key": "myprefix/testkey"})

        with stubber:
            result = cache.get(key)
            assert result == value
            stubber.assert_no_pending_responses()

    @patch("time.time", return_value=1000)
    def test_get_with_stale_data(self, mock_time):
        s3_client = boto3.client("s3")
        stubber = Stubber(s3_client)
        s3_path = "s3://mybucket/myprefix"
        cache = S3Cache(s3_path, freshness_in_seconds=50)
        cache._s3_client = s3_client

        key = "testkey"
        value = "testvalue"

        raw_response = json.dumps({"value": value, "cached_at": 900})
        encoded_message = raw_response.encode()
        raw_stream = StreamingBody(io.BytesIO(encoded_message), len(encoded_message))
        response = {"Body": raw_stream}
        stubber.add_response("get_object", response, {"Bucket": "mybucket", "Key": "myprefix/testkey"})

        with stubber:
            result = cache.get(key)
            assert result is None
            stubber.assert_no_pending_responses()

    def test_miss(self):
        s3_client = boto3.client("s3")
        stubber = Stubber(s3_client)
        s3_path = "s3://mybucket/myprefix"
        cache = S3Cache(s3_path, freshness_in_seconds=50)
        cache._s3_client = s3_client

        key = "testkey"

        stubber.add_client_error(
            "get_object",
            service_error_code="NoSuchKey",
            expected_params={"Bucket": "mybucket", "Key": "myprefix/testkey"},
        )

        with stubber:
            result = cache.get(key)
            assert result is None
            stubber.assert_no_pending_responses()

    @patch("time.time", return_value=1000)
    def test_set_with_freshness(self, mock_time):
        s3_client = boto3.client("s3")
        stubber = Stubber(s3_client)
        s3_path = "s3://mybucket/myprefix"
        cache = S3Cache(s3_path, freshness_in_seconds=50)
        cache._s3_client = s3_client

        key = "testkey"
        value = {"keyA": "a", "keyB": "b", "keyC": {"keyC.D": "d"}}

        params = {
            "Body": json.dumps({"value": value, "cached_at": 1000}, sort_keys=True, indent=2),
            "Bucket": "mybucket",
            "Key": "myprefix/testkey",
        }
        stubber.add_response("put_object", service_response={}, expected_params=params)

        with stubber:
            result = cache.set(key, value)
            assert result is None
            stubber.assert_no_pending_responses()
