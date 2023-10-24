from pathlib import Path
from typing import Callable

import pytest
from ray.data import Dataset

from sycamore.data import Document
from sycamore.transforms.partition import (
    Partition,
    HtmlPartitioner,
    UnstructuredPdfPartitioner,
    UnstructuredPPTXPartitioner,
)
from sycamore.scans import BinaryScan
from sycamore.tests.config import TEST_DIR


def _make_scan_executor(path: Path, format: str) -> Callable[[], Dataset]:
    def do_scan() -> Dataset:
        return BinaryScan(str(path), binary_format=format).execute()

    return do_scan


class TestPartition:
    def test_partitioner(self):
        dict = {
            "type": "Title",
            "coordinates": {
                "points": (
                    (116.519, 70.34515579999993),
                    (116.519, 100.63135580000005),
                    (481.02724959999995, 100.63135580000005),
                    (481.02724959999995, 70.34515579999993),
                ),
                "coordinate_system": "PixelSpace",
                "layout_width": 595.276,
                "layout_height": 841.89,
            },
            "element_id": "af2a328be129ce50f85b7946c35d1cf1",
            "metadata": {"filename": "Bert.pdf", "filetype": "application/pdf", "page_number": 1},
            "text": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        }
        element = UnstructuredPdfPartitioner.to_element(dict)
        assert element.type == "Title"
        assert (
            element.text_representation == "BERT: Pre-training of Deep Bidirectional Transformers for"
            " Language Understanding"
        )
        assert element.bbox.coordinates == (
            0.1957394553114858,
            0.08355623157419607,
            0.8080743211552288,
            0.11953028994286671,
        )
        assert element.properties == {
            "element_id": "af2a328be129ce50f85b7946c35d1cf1",
            "filename": "Bert.pdf",
            "filetype": "application/pdf",
            "page_number": 1,
        }

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, partition_count",
        [(UnstructuredPdfPartitioner(), TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)],
        indirect=["read_local_binary"],
    )
    def test_pdf_partitioner(self, partitioner, read_local_binary, partition_count):
        document = partitioner.partition(read_local_binary)
        assert len(document.elements) == partition_count

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, expected_partition_count",
        [
            (
                HtmlPartitioner(),
                TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html",
                76,
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
                77,
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

    @pytest.mark.parametrize("path, partition_count", [(TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)])
    def test_partition_pdf(self, mocker, path, partition_count) -> None:
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan, partitioner=UnstructuredPdfPartitioner())
        execute: Callable[[], Dataset] = _make_scan_executor(path, "pdf")
        mocker.patch.object(scan, "execute", execute)
        docset = partition.execute()
        doc = Document.from_row(docset.take(limit=1)[0])
        assert len(doc.elements) == partition_count

    @pytest.mark.parametrize(
        "path, partition_count", [(TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html", 76)]
    )
    def test_partition_html(self, mocker, path, partition_count) -> None:
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan, partitioner=HtmlPartitioner())
        execute: Callable[[], Dataset] = _make_scan_executor(path, "html")
        mocker.patch.object(scan, "execute", execute)
        docset = partition.execute()
        doc = Document.from_row(docset.take(limit=1)[0])
        assert len(doc.elements) == partition_count

    @pytest.mark.parametrize("path, partition_count", [(TEST_DIR / "resources/data/pptx/design.pptx", 71)])
    def test_partition_pptx(self, mocker, path, partition_count) -> None:
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan, partitioner=UnstructuredPPTXPartitioner())
        execute: Callable[[], Dataset] = _make_scan_executor(path, "pptx")
        mocker.patch.object(scan, "execute", execute)
        docset = partition.execute()
        doc = Document.from_row(docset.take(limit=1)[0])
        assert len(doc.elements) == partition_count
