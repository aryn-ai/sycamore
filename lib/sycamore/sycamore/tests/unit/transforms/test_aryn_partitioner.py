from sycamore.transforms.aryn_partitioner import ArynPDFPartitioner
from sycamore.tests.config import TEST_DIR
from sycamore.data.element import create_element
from sycamore.utils.deep_eq import assert_deep_eq
import json
import base64


class MockResponseNoTables:
    def __init__(self) -> None:
        self.status_code = 200

    def json(self) -> dict:
        path = TEST_DIR / "resources/data/json/model_server_output_transformer.json"
        return json.loads(open(str(path), "r").read())

    def text(self) -> str:
        return ""


class MockResponseTables:
    def __init__(self) -> None:
        self.status_code = 200

    def json(self) -> dict:
        path = TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        return json.loads(open(str(path), "r").read())

    def text(self) -> str:
        return ""


class TestArynPDFPartitioner:
    def test_partition(self, mocker) -> None:
        mocker.patch("requests.post", return_value=MockResponseNoTables())
        with open(TEST_DIR / "resources/data/json/model_server_output_transformer.json") as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())

                expected_elements = []
                for element_json in expected_json:
                    element = create_element(**element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(ArynPDFPartitioner.partition_pdf(pdf, aryn_token=""), expected_elements, [])

    def test_partition_extract_table_structure(self, mocker) -> None:
        mocker.patch("requests.post", return_value=MockResponseTables())
        with open(
            TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        ) as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Ray.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())

                expected_elements = []
                for element_json in expected_json:
                    element = create_element(**element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(
                    ArynPDFPartitioner.partition_pdf(pdf, extract_table_structure=True, aryn_token=""),
                    expected_elements,
                    [],
                )
