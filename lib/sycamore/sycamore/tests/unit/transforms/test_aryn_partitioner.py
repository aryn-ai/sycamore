import json
from sycamore.transforms.aryn_partitioner import ArynPDFPartitioner
from sycamore.tests.config import TEST_DIR


class TestArynPDFPartitioner:
    def test_partition(self):
        with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
            results = ArynPDFPartitioner.partition_pdf(pdf, aryn_token="")

            for element in results:
                json.dumps(element.properties)

    def test_partition_extract_table_structure(self):
        with open(TEST_DIR / "resources/data/pdfs/Transformer.pdf", "rb") as pdf:
            results = ArynPDFPartitioner.partition_pdf(pdf, aryn_token="", extract_table_structure=True)

            for element in results:
                json.dumps(element.properties)
