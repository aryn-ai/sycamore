from sycamore.rules import EnforceResourceUsage
from sycamore.scans import BinaryScan
from sycamore.transforms import Partition, Explode
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.writers import OpenSearchWriter


class TestRewriter:
    def test_enforce_resource_usage(self):
        scan = BinaryScan("path", binary_format="pdf")
        partition = Partition(scan, UnstructuredPdfPartitioner())
        explode = Explode(partition)
        writer = OpenSearchWriter(explode, "test", os_client_args={"a": 1, "b": "str"})

        rule = EnforceResourceUsage()
        writer.traverse_down(rule)
        assert scan.resource_args["num_cpus"] == 1 and scan.resource_args["num_gpus"] == 0
        assert partition.resource_args["num_cpus"] == 1
        assert explode.resource_args["num_cpus"] == 1 and explode.resource_args["num_gpus"] == 0
        assert writer.resource_args["num_cpus"] == 1 and writer.resource_args["num_gpus"] == 0
