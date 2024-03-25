import datetime
import os.path
from typing import Any

import boto3
import botocore.client
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
        if os.path.exists("/.dockerenv"):
            s = os.stat("/app/.data/.s3")
            if s.st_uid != 1000 or s.st_gid != 1000:
                raise RuntimeError(
                    f"Incorrect ownership on /app/.data/.s3 {s.st_uid},{s.st_gid}\nReset the containers or manually chown the files"
                )

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
                    print("WARNING, ignoring directory-like", object_key)

    # TODO: Parth - Rewrite this using s3.get_object API instead and handle the content type.
    def _download_if_new_object(self, object_metadata: dict[str, Any]) -> None:
        object_key = object_metadata["Key"]
        file_path = os.path.join(
            self._file_storage_location, self._get_file_extension(object_key), self._get_file_name(object_key)
        )
        try:
            if self._is_new_object(object_metadata, file_path):
                print("Downloading", object_key, "as", file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                self._s3_client.download_file(self._bucket_location, object_key, file_path)
            else:
                # print("Skipping up-to-date", object_key)
                pass
        except Exception:
            raise

    def _get_file_name(self, object_key: str) -> str:
        return object_key.replace("/", "_")

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
            last_modified_os = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), datetime.timezone.utc)
            last_modified_s3 = object_metadata.get("LastModified")
            if last_modified_s3 is None:
                print("WARNING: missing LastModified on object metadata, assuming newer", object_metadata["Key"])
                return False

            return last_modified_os < last_modified_s3
        except FileNotFoundError:
            # TODO: parth - if I change this to return False, no test fails
            return True

    def _get_s3_client(
        self, boto_session_args: list[Any], boto_session_kwargs: dict[str, Any]
    ) -> botocore.client.BaseClient:
        session = boto3.session.Session(*boto_session_args, **boto_session_kwargs)
        return session.client("s3")


if __name__ == "__main__":
    print("Version-Info, Sycamore Crawler S3 Branch:", os.environ.get("GIT_BRANCH", "unset"))
    print("Version-Info, Sycamore Crawler S3 Commit:", os.environ.get("GIT_COMMIT", "unset"))
    print("Version-Info, Sycamore Crawler S3 Diff:", os.environ.get("GIT_DIFF", "unset"))

    if len(sys.argv) > 3 or (len(sys.argv) > 1 and sys.argv[1] == "-h"):
        print("Usage : poetry run python s3_crawler.py bucket_name prefix_value")
    else:
        if len(sys.argv) == 1:
            bucket = "aryn-public"
            prefix = "sort-benchmark"
        elif len(sys.argv) == 2:
            bucket = sys.argv[1]
            prefix = ""
        else:
            bucket = sys.argv[1]
            prefix = sys.argv[2]

        s3 = S3Crawler(bucket, prefix)
        s3.crawl()
