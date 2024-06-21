from typing import BinaryIO
from collections.abc import Mapping
import requests
import json
from sycamore.data.element import create_element, Element
import time
from typing import List
import base64

_DEFAULT_ARYN_PARTITIONER_ADDRESS = "https://api.aryn.cloud/v1/document/partition"
_ARYN_PARTITIONING_SERVICE_WAIT_MESSAGE = '{"detail":"Please try again in a little while."}'
_ARYN_PARTITIONER_MAX_RETRIES = 6


class ArynPDFPartitionerException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ArynPDFPartitioner:
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
        max_retries: int = _ARYN_PARTITIONER_MAX_RETRIES,
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

        last_status_code = 500
        last_message = _ARYN_PARTITIONING_SERVICE_WAIT_MESSAGE
        tries = 0
        while (
            last_status_code == 500 and last_message == _ARYN_PARTITIONING_SERVICE_WAIT_MESSAGE and tries < max_retries
        ):
            response = requests.post(aryn_partitioner_address, files=files, headers=header)
            last_status_code = response.status_code
            last_message = response.text
            if tries > 0:
                time.sleep(10 * (2**tries))  # Modified binary exponential backoff
            tries += 1

        if response.status_code != 200:
            raise ArynPDFPartitionerException(f"Error: status_code: {response.status_code}, reason: {response.text}")

        response_json = response.json()

        elements = []
        for element_json in response_json:
            element = create_element(**element_json)
            if element.binary_representation:
                element.binary_representation = base64.b64decode(element.binary_representation)
            elements.append(element)

        return elements
