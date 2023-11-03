import datetime
import os.path
from typing import Any

import boto3
import botocore.client
from dateutil.tz import tzutc
import sys


class S3Crawler:
    def __init__(
        self,
        bucket_location: str,
        prefix: str,
        boto_session_args: list[Any] = [],
        boto_session_kwargs: dict[str, Any] = {},
    ):
        self._bucket_location = bucket_location
        self._prefix = prefix
        self._file_storage_location = "./.data/.s3/downloads"
        self._s3_client = self._get_s3_client(boto_session_args, boto_session_kwargs)

    def crawl(self) -> None:
        self._find_and_download_new_objects()

    def _find_and_download_new_objects(self) -> None:
        paginator = self._s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self._bucket_location, Prefix=self._prefix)

        for page in page_iterator:
            for s3_object in page["Contents"]:
                object_key = s3_object["Key"]
                if not object_key.endswith("/"):
                    self._download_if_new_object(s3_object)
                else:
                    print("WARNING, skipping", object_key)

    # TODO: Parth - Rewrite this using s3.get_object API instead and handle the content type.
    def _download_if_new_object(self, object_metadata: dict[str, Any]) -> None:
        object_key = object_metadata["Key"]
        file_path = os.path.join(
            self._file_storage_location, self._get_file_extension(object_key), self._get_file_name(object_key)
        )
        try:
            if self._is_new_object(object_metadata, file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                self._s3_client.download_file(self._bucket_location, object_key, file_path)
        except Exception:
            raise

    def _get_file_name(self, object_key: str) -> str:
        return object_key.split("/")[1]

    def _get_file_extension(self, object_key: str) -> str:
        extension = os.path.splitext(object_key)[1][1:]
        if extension == "pdf":
            return "pdf"
        elif extension == "html":
            return "html"
        else:
            return "unknown"

    def _is_new_object(self, object_metadata: dict[str, Any], file_path: str) -> bool:
        try:
            last_modified_s3: datetime.datetime = object_metadata.get("LastModified")
            last_modified_os = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), tzutc())

            return last_modified_os < last_modified_s3
        except FileNotFoundError:
            return True

    def _get_s3_client(
        self, boto_session_args: list[Any], boto_session_kwargs: dict[str, Any]
    ) -> botocore.client.BaseClient:
        session = boto3.session.Session(*boto_session_args, **boto_session_kwargs)
        return session.client("s3")


if __name__ == "__main__":
    if len(sys.argv) > 3 or sys.argv[1] == "-h":
        print("Usage : poetry run python s3_crawler.py bucket_name prefix_value")
    else:
        if len(sys.argv) == 1:
            bucket = "aryn-public"
            prefix = "sort_benchmark"
        elif len(sys.argv) == 2:
            bucket = sys.argv[1]
            prefix = ""
        else:
            bucket = sys.argv[1]
            prefix = sys.argv[2]

        s3 = S3Crawler(bucket, prefix)
        s3.crawl()
