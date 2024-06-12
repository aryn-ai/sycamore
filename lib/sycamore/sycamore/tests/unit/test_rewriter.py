from sycamore.rules import EnforceResourceUsage
from sycamore.scans import BinaryScan
from sycamore.transforms import Partition, Explode
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.writers import OpenSearchWriter
from sycamore.writers.opensearch import OpenSearchClientParams, OpenSearchTargetParams


class TestRewriter:
    def test_enforce_resource_usage(self):
        scan = BinaryScan("path", binary_format="pdf")
        partition = Partition(scan, UnstructuredPdfPartitioner())
        explode = Explode(partition)
        writer = OpenSearchWriter(explode, OpenSearchClientParams(), OpenSearchTargetParams(index_name="test"))

        rule = EnforceResourceUsage()
        writer.traverse_down(rule)
        assert scan.resource_args["num_cpus"] == 1 and "num_gpus" not in scan.resource_args
        assert explode.resource_args["num_cpus"] == 1 and "num_gpus" not in explode.resource_args
        assert writer.resource_args["num_cpus"] == 1 and "num_gpus" not in writer.resource_args
