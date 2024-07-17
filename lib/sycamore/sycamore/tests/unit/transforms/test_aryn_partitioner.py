from sycamore.transforms.detr_partitioner import ArynPDFPartitioner
from sycamore.tests.config import TEST_DIR
from sycamore.data.element import create_element
from sycamore.utils.deep_eq import assert_deep_eq
from sycamore.utils.http import OneShotKaClient
import json
import base64


class MockResponse:
    status = 200

    def init(self):
        pass

    def getheaders(self):
        return {}


class MockOneShotKaClient:
    def __init__(self) -> None:
        self.resp = MockResponse()

    def post(self, headers, form) -> bytes:
        path = TEST_DIR / "resources/data/json/model_server_output_transformer.json"
        return open(str(path), "rb").read()


class MockOneShotKaClientTables:
    def __init__(self) -> None:
        self.resp = MockResponse()

    def post(self, headers, form) -> bytes:
        path = TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        return open(str(path), "rb").read()


class TestArynPDFPartitioner:
    def test_partition(self, mocker) -> None:
        mocker.patch.object(OneShotKaClient, "__new__", return_value=MockOneShotKaClient())
        with open(TEST_DIR / "resources/data/json/model_server_output_transformer.json") as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())
                partitioner = ArynPDFPartitioner()
                expected_elements = []
                for element_json in expected_json:
                    element = create_element(**element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(partitioner.partition_pdf(pdf, aryn_api_key=""), expected_elements, [])

    def test_partition_extract_table_structure(self, mocker) -> None:
        mocker.patch.object(OneShotKaClient, "__new__", return_value=MockOneShotKaClientTables())
        with open(
            TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        ) as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())
                partitioner = ArynPDFPartitioner()
                expected_elements = []
                for element_json in expected_json:
                    element = create_element(**element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(
                    partitioner.partition_pdf(pdf, extract_table_structure=True, aryn_api_key=""),
                    expected_elements,
                    [],
                )
