import json
from sycamore.transforms.aryn_partitioner import ArynPDFPartitioner
from sycamore.tests.config import TEST_DIR
import requests

class MockResponse:
    def __init__(self):
        self.status_code = 200
    def json(self):
        return {}

class TestArynPDFPartitioner:
    def test_partition(self, mocker):
        mocker.patch("requests.post", return_value=MockResponse())
        with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
            results = ArynPDFPartitioner.partition_pdf(pdf, aryn_token="")
        

    def test_partition_extract_table_structure(self, mocker):
        mocker.patch("requests.post", return_value=MockResponse())
        with open(TEST_DIR / "resources/data/pdfs/Transformer.pdf", "rb") as pdf:
            results = ArynPDFPartitioner.partition_pdf(pdf, aryn_token="", extract_table_structure=True)

