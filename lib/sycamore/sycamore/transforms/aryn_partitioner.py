from typing import BinaryIO
from collections.abc import Mapping
import requests
import json
from sycamore.data.element import create_element

_DEFAULT_ARYN_PARTITIONER_ADDRESS = "https://api.aryn.cloud:8000/v1/partition"


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
    ):
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
            raise ArynPDFPartitionerException(f"Error: status_code: {response.status_code}, reason: {response.text}")

        response_json = response.json()

        elements = []
        for element_json in response_json:
            element = create_element(**element_json)
            elements.append(element)

        return elements
