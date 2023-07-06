import pytest
from ray.data import read_binary_files, Dataset
from shannon.execution.scans import BinaryScan
from shannon.execution.transforms import PartitionPDF
from shannon.tests.config import TEST_DIR
from typing import Callable


class TestPartition:
    @pytest.mark.parametrize(
        "path, partition_count",
        [(TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)])
    def test_pdf_partition(self, mocker, path, partition_count):
        scan = mocker.Mock(spec=BinaryScan)
        partition = PartitionPDF(scan, col_name="bytes")
        execute: Callable[[], Dataset] = lambda: read_binary_files(path)
        mocker.patch.object(scan, "execute", execute)
        dataset = partition.execute()
        assert (len(dataset.take(limit=1000)) == partition_count)
