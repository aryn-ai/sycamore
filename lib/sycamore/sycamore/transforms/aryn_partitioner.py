from typing import BinaryIO
from collections.abc import Mapping
import requests
import json
from sycamore.data.element import create_element, Element
from typing import List
import base64
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_delay

_DEFAULT_ARYN_PARTITIONER_ADDRESS = "https://api.aryn.cloud/v1/document/partition"
_ARYN_PARTITIONING_SERVICE_WAIT_MESSAGE = '{"detail":"Please try again in a little while."}'
_TEN_MINUTES = 600


class ArynPDFPartitionerException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ArynContinuePDFPartitionerException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ArynPDFPartitioner:
    @retry(
        retry=retry_if_exception_type(ArynContinuePDFPartitionerException),
        wait=wait_exponential(multiplier=1, min=1),
        stop=stop_after_delay(_TEN_MINUTES),
    )
    @staticmethod
    def partition_pdf(
        file: BinaryIO,
        aryn_token: str,
        aryn_partitioner_address=_DEFAULT_ARYN_PARTITIONER_ADDRESS,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_tables: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
    ) -> List[Element]:
        options = {
            "threshold": threshold,
            "use_ocr": use_ocr,
            "ocr_images": ocr_images,
            "ocr_tables": ocr_tables,
            "extract_table_structure": extract_table_structure,
            "extract_images": extract_images,
        }

        files: Mapping = {"pdf": file, "options": json.dumps(options).encode("utf-8")}
        header = {"Authorization": f"Bearer {aryn_token}"}

        response = requests.post(aryn_partitioner_address, files=files, headers=header)

        if response.status_code != 200:
            if response.status_code == 500 and response.text == _ARYN_PARTITIONING_SERVICE_WAIT_MESSAGE:
                raise ArynContinuePDFPartitionerException(
                    f"Error: status_code: {response.status_code}, reason: {response.text}"
                )
            raise ArynPDFPartitionerException(f"Error: status_code: {response.status_code}, reason: {response.text}")

        response_json = response.json()

        elements = []
        for element_json in response_json:
            element = create_element(**element_json)
            if element.binary_representation:
                element.binary_representation = base64.b64decode(element.binary_representation)
            elements.append(element)

        return elements
