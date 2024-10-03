from sycamore.transforms.detr_partitioner import ArynPDFPartitioner
from sycamore.tests.config import TEST_DIR
from sycamore.data.element import create_element
from sycamore.utils.deep_eq import assert_deep_eq
import json
import base64


class MockResponseNoTables:
    def __init__(self) -> None:
        self.status_code = 200

    def iter_content(self, chunksize):
        path = TEST_DIR / "resources/data/json/model_server_output_transformer.json"
        yield open(str(path), "rb").read()


class MockResponseTables:
    def __init__(self) -> None:
        self.status_code = 200

    def iter_content(self, chunksize):
        path = TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        yield open(str(path), "rb").read()


class TestArynPDFPartitioner:
    def test_partition(self, mocker) -> None:
        mocker.patch("requests.post", return_value=MockResponseNoTables())
        with open(TEST_DIR / "resources/data/json/model_server_output_transformer.json") as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Transformer.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())
                partitioner = ArynPDFPartitioner(None)
                expected_elements = []
                for i, element_json in enumerate(expected_json):
                    element = create_element(i, **element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(partitioner.partition_pdf(pdf, aryn_api_key="mocked"), expected_elements, [])

    def test_partition_extract_table_structure(self, mocker) -> None:
        mocker.patch("requests.post", return_value=MockResponseTables())
        with open(
            TEST_DIR / "resources/data/json/model_server_output_transformer_extract_tables.json"
        ) as expected_text:
            with open(TEST_DIR / "resources/data/pdfs/Transformer.pdf", "rb") as pdf:
                expected_json = json.loads(expected_text.read())
                partitioner = ArynPDFPartitioner(None)
                expected_elements = []
                for i, element_json in enumerate(expected_json):
                    element = create_element(i, **element_json)
                    if element.binary_representation:
                        element.binary_representation = base64.b64decode(element.binary_representation)
                    expected_elements.append(element)

                assert_deep_eq(
                    partitioner.partition_pdf(pdf, extract_table_structure=True, aryn_api_key="mocked"),
                    expected_elements,
                    [],
                )
