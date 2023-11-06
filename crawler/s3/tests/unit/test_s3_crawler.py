from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

import botocore.paginate

from crawler.s3.crawler import s3_crawler
from crawler.s3.crawler.s3_crawler import S3Crawler


# TODO : Parth - investigate doing an integration test similar to the http one. Maybe using minio or localstack
class TestS3Crawler:
    def setup_mocks(
        self,
        mocker,
        crawler: S3Crawler,
        mock_page_iterator: list[dict[str, Any]],
        downloaded_files: list[str],
    ):
        paginator = mocker.patch.object(crawler._s3_client, "get_paginator")
        mock_paginator = mocker.Mock(spec=botocore.paginate.Paginator)
        paginator.return_value = mock_paginator

        paginate = mocker.patch.object(mock_paginator, "paginate")
        paginate.return_value = mock_page_iterator

        mock_os = mocker.patch.object(s3_crawler.os.path, "getmtime")
        mock_os.return_value = (datetime.now(UTC) - timedelta(1)).timestamp()

        def mock_download_func(bucket, key, file):
            downloaded_files.append(key)

        download = mocker.patch.object(crawler._s3_client, "download_file")
        download.side_effect = mock_download_func

    def test_s3_crawler_first_crawl(self, mocker, tmp_path: Path):
        s3_crawler = S3Crawler("bucket", "prefix")
        mock_page_iterator = [
            {"Contents": [{"Key": "folder/key1", "Etag": "etag1", "LastModified": datetime.now(UTC), "Size": "0"}]}
        ]

        downloaded_objects: list[str] = []
        self.setup_mocks(mocker, s3_crawler, mock_page_iterator, downloaded_objects)

        s3_crawler.crawl()
        assert len(mock_page_iterator[0]["Contents"]) == len(downloaded_objects)

    def test_s3_crawler_finds_new_object(self, mocker, tmp_path: Path):
        s3_crawler = S3Crawler("bucket", "prefix")
        mock_page_iterator = [
            {
                "Contents": [
                    {
                        "Key": "folder/key1.pdf",
                        "Etag": "etag1",
                        "LastModified": datetime.now(UTC) + timedelta(1),
                        "Size": "0",
                    },
                    {
                        "Key": "folder/key2.pdf",
                        "Etag": "etag2",
                        "LastModified": datetime.now(UTC) - timedelta(2),
                        "Size": "0",
                    },
                ]
            }
        ]

        downloaded_objects: list[str] = []
        self.setup_mocks(mocker, s3_crawler, mock_page_iterator, downloaded_objects)

        s3_crawler.crawl()
        assert len(mock_page_iterator[0]["Contents"]) == len(downloaded_objects) + 1

    def test_s3_crawler_nothing_to_crawl(self, mocker, tmp_path: Path):
        s3_crawler = S3Crawler("bucket", "prefix")

        mock_page_iterator = [
            {
                "Contents": [
                    {
                        "Key": "folder/key1",
                        "Etag": "etag1",
                        "LastModified": datetime.now(UTC) - timedelta(2),
                        "Size": "0",
                    },
                    {
                        "Key": "folder/key2",
                        "Etag": "etag2",
                        "LastModified": datetime.now(UTC) - timedelta(2),
                        "Size": "0",
                    },
                ]
            }
        ]

        downloaded_objects: list[str] = []
        self.setup_mocks(mocker, s3_crawler, mock_page_iterator, downloaded_objects)

        s3_crawler.crawl()
        assert len(downloaded_objects) == 0
