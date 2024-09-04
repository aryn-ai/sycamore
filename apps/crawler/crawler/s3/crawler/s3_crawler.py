import argparse
import datetime
import os.path
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from mypy_boto3_s3.client import S3Client
import sys


def usage(val: int) -> None:
    print("Usage: poetry run python s3_crawler.py bucket_name prefix_value")
    sys.exit(val)


class S3Crawler:
    def __init__(
        self,
        bucket_location: str,
        prefix: str,
        anon: bool,
        boto_session_args: list[Any] = [],
        boto_session_kwargs: dict[str, Any] = {},
    ):
        self._bucket_location = bucket_location
        self._prefix = prefix
        self._anon = anon
        self._session = boto3.session.Session(*boto_session_args, **boto_session_kwargs)
        self._file_storage_location = "./.data/.s3/downloads"
        if os.path.exists("/.dockerenv"):
            s = os.stat("/app/.data/.s3")
            if s.st_uid != 1000 or s.st_gid != 1000:
                raise RuntimeError(
                    f"Incorrect ownership on /app/.data/.s3 {s.st_uid},{s.st_gid}"
                    "\nTo fix: docker compose run fixuser"
                )

    def crawl(self) -> None:
        try:
            self._s3_client = self._get_s3_client()
            self._find_and_download_new_objects()
            return
        except NoCredentialsError:
            if self._anon:
                raise
            print("Automatically retrying in anonymous mode")
            self._anon = True
            self._s3_client = self._get_s3_client()
            self._find_and_download_new_objects()

    def _find_and_download_new_objects(self) -> None:
        paginator = self._s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self._bucket_location, Prefix=self._prefix)

        for page in page_iterator:
            for s3_object in page["Contents"]:
                object_key = s3_object["Key"]
                if not object_key.endswith("/"):
                    self._download_if_new_object(s3_object)  # type: ignore
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

    def _get_s3_client(self) -> S3Client:
        if self._anon:
            cfg = Config(signature_version=UNSIGNED)
            return self._session.client("s3", config=cfg)
        else:
            return self._session.client("s3")


if __name__ == "__main__":
    print("Version-Info, Sycamore Crawler S3 Branch:", os.environ.get("GIT_BRANCH", "unset"))
    print("Version-Info, Sycamore Crawler S3 Commit:", os.environ.get("GIT_COMMIT", "unset"))
    print("Version-Info, Sycamore Crawler S3 Diff:", os.environ.get("GIT_DIFF", "unset"))

    parser = argparse.ArgumentParser(
        description="The Sycamore crawler for Amazon AWS S3, from Aryn.ai",
    )
    parser.add_argument("bucket", nargs="?", default="", help="The AWS S3 bucket to crawl")
    parser.add_argument("prefix", nargs="?", default="", help="The prefix within the bucket to crawl")
    parser.add_argument("--anon", action="store_true", help="For accessing public buckets without credentials")
    args = parser.parse_args()

    # We'd like to auto-detect when no credentials are present, instead of
    # using --anon, but boto3 is clever finding creds and does so very late.

    if not args.bucket:
        args.bucket = "aryn-public"
        args.prefix = "sort-benchmark"

    s3 = S3Crawler(args.bucket, args.prefix, args.anon)
    s3.crawl()
