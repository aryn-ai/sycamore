from pathlib import Path
from typing import Callable

import pytest
from ray.data import Dataset

from sycamore.data import Document, Element
from sycamore.transforms.partition import (
    Partition,
    HtmlPartitioner,
    UnstructuredPdfPartitioner,
    UnstructuredPPTXPartitioner,
    SycamorePartitioner,
)
from sycamore.scans import BinaryScan
from sycamore.tests.config import TEST_DIR

import torch


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="SycamorePartitioner requires CUDA")
    @pytest.mark.skip(reason="Model File is not available")
    @pytest.mark.parametrize("path, partition_count", [(TEST_DIR / "resources/data/pdfs/Ray.pdf", 267)])
    def test_deformable_detr_partition(self, mocker, path, partition_count) -> None:
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan, partitioner=SycamorePartitioner(""))
        execute: Callable[[], Dataset] = _make_scan_executor(path, "pdf")
        mocker.patch.object(scan, "execute", execute)
        docset = partition.execute()
        doc = Document.from_row(docset.take(limit=1)[0])
        assert len(doc.elements) == partition_count

    def test_sycamore_partitioner_elements_reorder(self) -> None:
        import functools

        # e1.y1 < e0.y1 = e2.y1, e0.x1 < e2.x1 both on left
        e0 = Element({"bbox": (0.20, 0.50, 0.59, 0.59), "properties": {"page_number": 3}})
        e1 = Element({"bbox": (0.20, 0.21, 0.59, 0.59), "properties": {"page_number": 3}})
        e2 = Element({"bbox": (0.40, 0.50, 0.59, 0.59), "properties": {"page_number": 3}})

        # e4, e5 in left column, e4.y < e5.y1; e3, e6 in right columns, e3.y1 < e6.y1
        e3 = Element({"bbox": (0.60, 0.21, 0.59, 0.59), "properties": {"page_number": 1}})
        e4 = Element({"bbox": (0.20, 0.21, 0.59, 0.59), "properties": {"page_number": 1}})
        e5 = Element({"bbox": (0.20, 0.71, 0.59, 0.59), "properties": {"page_number": 1}})
        e6 = Element({"bbox": (0.70, 0.51, 0.59, 0.59), "properties": {"page_number": 1}})

        # all the same, test stable
        e7 = Element({"bbox": (0.20, 0.21, 0.59, 0.59), "properties": {"page_number": 2}})
        e8 = Element({"bbox": (0.20, 0.21, 0.59, 0.59), "properties": {"page_number": 2}})
        e9 = Element({"bbox": (0.20, 0.21, 0.59, 0.59), "properties": {"page_number": 2}})

        elements = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
        elements.sort(key=functools.cmp_to_key(SycamorePartitioner._elements_reorder))
        result = [e4, e5, e3, e6, e7, e8, e9, e1, e0, e2]

        assert elements == result

    def test_simple_ocr(self):
        import pdf2image
        from sycamore.transforms.detr_partitioner import extract_ocr

        path = TEST_DIR / "resources/data/ocr_pdfs/test_simple_ocr.pdf"
        images = pdf2image.convert_from_path(path, dpi=800)
        assert len(images) == 1

        elem = Element({"bbox": (0.0, 0.0, 1.0, 1.0), "properties": {"page_number": 1}})

        new_elems = extract_ocr(images, [[elem]])

        assert len(new_elems) == 1
        assert len(new_elems[0]) == 1

        text = new_elems[0][0].text_representation
        assert text is not None
        assert text.strip() == "The quick brown fox"
