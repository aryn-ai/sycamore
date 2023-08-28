import pytest

from sycamore.data import Element
from sycamore.execution.transforms.html.html_partitioner import HtmlPartitioner
from sycamore.tests.config import TEST_DIR


class TestHtmlPartition:
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
