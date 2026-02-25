from pathlib import Path
from typing import Callable

import pytest
from ray.data import Dataset

from sycamore.data import Document
from sycamore.transforms.partition import (
    Partition,
    HtmlPartitioner,
)
from sycamore.connectors.file import BinaryScan
from sycamore.tests.config import TEST_DIR


def _make_scan_executor(path: Path, format: str) -> Callable[[], Dataset]:
    def do_scan(**kwargs) -> Dataset:
        return BinaryScan(str(path), binary_format=format).execute(**kwargs)

    return do_scan


class TestPartition:

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, expected_partition_count",
        [
            (
                HtmlPartitioner(),
                TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html",
                73,
            )
        ],
        indirect=["read_local_binary"],
    )
    def test_html_partitioner(self, partitioner, read_local_binary, expected_partition_count):
        document = partitioner.partition(read_local_binary)
        assert len(document["elements"]) == expected_partition_count

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, expected_partition_count, expected_table_count",
        [
            (
                HtmlPartitioner(extract_tables=True),
                TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html",
                74,
                1,
            )
        ],
        indirect=["read_local_binary"],
    )
    def test_html_partitioner_with_tables(
        self, partitioner, read_local_binary, expected_partition_count, expected_table_count
    ):
        document = partitioner.partition(read_local_binary)
        assert len(document["elements"]) == expected_partition_count
        table_count = 0
        for partition in document["elements"]:
            if partition["type"] == "table":
                table_count += 1
        assert expected_table_count == table_count

    @pytest.mark.parametrize(
        "path, partition_count", [(TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html", 73)]
    )
    def test_partition_html(self, mocker, path, partition_count) -> None:
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan, partitioner=HtmlPartitioner())
        execute: Callable[[], Dataset] = _make_scan_executor(path, "html")
        mocker.patch.object(scan, "execute", execute)
        docset = partition.execute()
        doc = Document.from_row(docset.take(limit=1)[0])
        assert len(doc.elements) == partition_count
