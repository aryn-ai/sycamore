from sycamore.scans import BinaryScan
from sycamore.tests.config import TEST_DIR


class TestBinaryScan:
    def test_partition(self):
        paths = str(TEST_DIR / "resources/data/pdfs/")
        scan = BinaryScan(paths, binary_format="pdf")
        ds = scan.execute()
        assert ds.schema().names == [
            "doc_id",
            "type",
            "text_representation",
            "binary_representation",
            "elements",
            "embedding",
            "parent_id",
            "properties",
        ]
