from shannon.tests.config import TEST_DIR
import pytest
from shannon.execution.kernels import UnstructuredPartitionPdfKernel


class TestPartition:
    @pytest.mark.parametrize(
        "partitioner, read_local_binary, partition_count",
        [(UnstructuredPartitionPdfKernel("bytes"),
          TEST_DIR / "resources/data/pdfs/Transformer.pdf", 254)],
        indirect=["read_local_binary"])
    def test_pdf_partition(
            self, partitioner, read_local_binary, partition_count):
        partitions = partitioner.partition(read_local_binary)
        assert (len(partitions) == partition_count)
