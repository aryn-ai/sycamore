from typing import Callable

import pytest
from ray.data import Dataset

from sycamore.execution.transforms.partition import HtmlPartitioner
from sycamore.execution.scans import BinaryScan
from sycamore.execution.transforms import Partition
from sycamore.execution.transforms.partition import \
    (Partitioner, PdfPartitioner)
from sycamore.tests.config import TEST_DIR


class TestPartition:
    def test_partitioner(self):
        dict = {
            'type': 'Title',
            'coordinates': (
                (116.519, 70.34515579999993), (116.519, 100.63135580000005),
                (481.02724959999995, 100.63135580000005),
                (481.02724959999995, 70.34515579999993)),
            'coordinate_system': 'PixelSpace',
            'layout_width': 595.276,
            'layout_height': 841.89,
            'element_id': 'af2a328be129ce50f85b7946c35d1cf1',
            'metadata': {
                'filename': 'Bert.pdf',
                'filetype': 'application/pdf',
                'page_number': 1},
            'text': 'BERT: Pre-training of Deep Bidirectional Transformers '
                    'for Language Understanding'}
        element = Partitioner.to_element(dict)
        assert (element.type == "Title")
        assert (element.content ==
                "BERT: Pre-training of Deep Bidirectional Transformers for"
                " Language Understanding")
        assert (element.properties == {
            'coordinates': (
                (116.519, 70.34515579999993), (116.519, 100.63135580000005),
                (481.02724959999995, 100.63135580000005),
                (481.02724959999995, 70.34515579999993)),
            'coordinate_system': 'PixelSpace',
            'layout_width': 595.276,
            'layout_height': 841.89,
            'element_id': 'af2a328be129ce50f85b7946c35d1cf1',
            'filename': 'Bert.pdf',
            'filetype': 'application/pdf',
            'page_number': 1})

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, partition_count",
        [(PdfPartitioner(),
          TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)],
        indirect=["read_local_binary"])
    def test_pdf_partitioner(
            self, partitioner, read_local_binary, partition_count):
        document = partitioner.partition(read_local_binary)
        assert (len(document["elements"]["array"]) == partition_count)


    @pytest.mark.parametrize(
        "partitioner, read_local_binary, expected_partition_count",
        [(HtmlPartitioner(),
          TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html", 76)],
        indirect=["read_local_binary"])
    def test_html_partitioner(
            self, partitioner, read_local_binary, expected_partition_count):
        document = partitioner.partition(read_local_binary)
        assert (len(document["elements"]["array"]) == expected_partition_count)

    @pytest.mark.parametrize(
        "partitioner, read_local_binary, expected_partition_count, expected_table_count",
        [(HtmlPartitioner(extract_tables=True),
          TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html", 77, 1)],
        indirect=["read_local_binary"])
    def test_html_partitioner_with_tables(
            self, partitioner, read_local_binary, expected_partition_count, expected_table_count):
        document = partitioner.partition(read_local_binary)
        assert (len(document["elements"]["array"]) == expected_partition_count)
        table_count = 0
        for partition in document["elements"]["array"]:
            if partition["type"] == "table":
                table_count += 1
        assert expected_table_count == table_count

    @pytest.mark.parametrize(
        "path, partition_count",
        [(TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)])
    def test_partition_pdf(self, mocker, path, partition_count):
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan)
        execute: Callable[[], Dataset] = \
            lambda: BinaryScan(path, binary_format="pdf").execute()
        mocker.patch.object(scan, "execute", execute)
        partition.partitioner = PdfPartitioner()
        docset = partition.execute()
        doc = docset.take(limit=1)[0]
        assert (len(doc["elements"]["array"]) == partition_count)

    @pytest.mark.parametrize(
        "path, partition_count",
        [(TEST_DIR / "resources/data/htmls/wikipedia_binary_search.html", 76)])
    def test_partition_html(self, mocker, path, partition_count):
        scan = mocker.Mock(spec=BinaryScan)
        partition = Partition(scan)
        execute: Callable[[], Dataset] = \
            lambda: BinaryScan(path, binary_format="html").execute()
        mocker.patch.object(scan, "execute", execute)
        partition.partitioner = HtmlPartitioner()
        docset = partition.execute()
        doc = docset.take(limit=1)[0]
        assert (len(doc["elements"]["array"]) == partition_count)

