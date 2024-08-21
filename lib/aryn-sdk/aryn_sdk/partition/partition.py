from typing import BinaryIO, Optional, Union
from collections.abc import Mapping
from aryn_sdk.config import ArynConfig
import requests
import sys
import json
import logging
import pandas as pd
import numpy as np
from collections import OrderedDict


# URL for Aryn Partitioning Service (APS)
APS_URL = "https://api.aryn.cloud/v1/document/partition"

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(sys.stderr))


def partition_file(
    file: BinaryIO,
    aryn_api_key: Optional[str] = None,
    aryn_config: ArynConfig = ArynConfig(),
    threshold: Optional[float] = None,
    use_ocr: bool = False,
    ocr_images: bool = False,
    ocr_model: str = "easy",
    extract_table_structure: bool = False,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    aps_url: str = APS_URL,
    ssl_verify: bool = True,
) -> dict:
    """
    Sends file to the Aryn Partitioning Service and returns a dict of its document structure and text

    Args:
        file: open pdf file to partition
        aryn_api_key: aryn api key, provided as a string
        aryn_config: ArynConfig object, used for finding an api key.
            If aryn_api_key is set it will override this.
            default: The default ArynConfig looks in the env var ARYN_API_KEY and the file ~/.aryn/config.yaml
        threshold:  value in [0.0 .. 1.0] to specify the cutoff for detecting bounding boxes.
            default: None (APS will choose)
        use_ocr: extract text using an OCR model instead of extracting embedded text in PDF.
            default: False
        ocr_images: attempt to use OCR to generate a text representation of detected images.
            default: False
        extract_table_structure: extract tables and their structural content.
            default: False
        extract_images: extract image contents.
            default: False
        selected_pages: list of individual pages (1-indexed) from the pdf to partition
            default: None
        aps_url: url of the Aryn Partitioning Service endpoint.
            default: "https://api.aryn.cloud/v1/document/partition"
        ssl_verify: verify ssl certificates. In databricks, set this to False to fix ssl imcompatibilities.

    Returns:
        A dictionary containing "status" and "elements"

    Example:
         .. code-block:: python

            from aryn_sdk.partition import partition_file

            with open("my-favorite-pdf.pdf", "rb") as f:
                data = partition_file(
                    f,
                    aryn_api_key="MY-ARYN-TOKEN",
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True
                )
            elements = data['elements']
    """

    if aryn_api_key is not None:
        if aryn_config is not None:
            _logger.warning("Both aryn_api_key and aryn_config were provided. Using aryn_api_key")
        aryn_config = ArynConfig(aryn_api_key=aryn_api_key)

    options_str = _json_options(
        threshold=threshold,
        use_ocr=use_ocr,
        ocr_images=ocr_images,
        ocr_model=ocr_model,
        extract_table_structure=extract_table_structure,
        extract_images=extract_images,
        selected_pages=selected_pages,
    )

    _logger.debug(f"{options_str}")

    files: Mapping = {"options": options_str.encode("utf-8"), "pdf": file}
    http_header = {"Authorization": "Bearer {}".format(aryn_config.api_key())}
    resp = requests.post(aps_url, files=files, headers=http_header, stream=True, verify=ssl_verify)

    if resp.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"Error: status_code: {resp.status_code}, reason: {resp.text}", response=resp
        )

    content = []
    partial_line = []
    in_bulk = False
    for part in resp.iter_content(None):
        if not part:
            continue

        content.append(part)
        if in_bulk:
            continue

        partial_line.append(part)
        if b"\n" not in part:
            continue

        these_lines = b"".join(partial_line).split(b"\n")
        partial_line = [these_lines.pop()]

        for line in these_lines:
            if line.startswith(b"  ],"):
                in_bulk = True
                break
            if line.startswith(b'    "T+'):
                t = json.loads(line.decode("utf-8").removesuffix(","))
                _logger.info(f"ArynPartitioner: {t}")
    body = b"".join(content).decode("utf-8")
    _logger.debug("Recieved data from ArynPartitioner")

    data = json.loads(body)
    assert isinstance(data, dict)
    status = data.get("status", [])
    if "error" in data:
        error_msg = (
            "Limit Exceeded:"
            if "Please try again in a little while" in data["error"]
            else "Error partway through processing:"
        )
        raise ValueError(f"{error_msg} {data['error']}\nPartial Status:\n{status}")
    return data


def _json_options(
    threshold: Optional[float] = None,
    use_ocr: bool = False,
    ocr_images: bool = False,
    ocr_model: str = "easy",
    extract_table_structure: bool = False,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
) -> str:
    # isn't type-checking fun
    options: dict[str, Union[float, bool, str, list[Union[list[int], int]]]] = dict()
    if threshold:
        options["threshold"] = threshold
    if use_ocr:
        options["use_ocr"] = use_ocr
    if ocr_images:
        options["ocr_images"] = ocr_images
    if ocr_model:
        options["ocr_model"] = ocr_model
    if extract_images:
        options["extract_images"] = extract_images
    if extract_table_structure:
        options["extract_table_structure"] = extract_table_structure
    if selected_pages:
        options["selected_pages"] = selected_pages

    options["source"] = "aryn-sdk"

    return json.dumps(options)


# Heavily adapted from lib/sycamore/data/table.py::Table.to_csv()
def table_elem_to_dataframe(elem: dict) -> Optional[pd.DataFrame]:
    """
    Create a pandas DataFrame representing the tabular data inside the provided table element.
    If the element is not of type 'table' or doesn't contain any table data, return None instead.

    Args:
        elem: An element from the 'elements' field of a ``partition_file`` response.

    Example:
         .. code-block:: python

            from aryn_sdk.partition import partition_file, table_elem_to_dataframe

            with open("partition-me.pdf", "rb") as f:
                data = partition_file(
                    f,
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True
                )

            # Find the first table and convert it to a dataframe
            df = None
            for element in data['elements']:
                if element['type'] == 'table':
                    df = table_elem_to_dataframe(element)
                    break
    """

    if (elem["type"] != "table") or (elem["table"] is None):
        return None

    table = elem["table"]

    header_rows = sorted(set(row_num for cell in table["cells"] for row_num in cell["rows"] if cell["is_header"]))
    i = -1
    for i in range(len(header_rows)):
        if header_rows[i] != i:
            break
    max_header_prefix_row = i
    grid_width = table["num_cols"]
    grid_height = table["num_rows"]

    grid = np.empty([grid_height, grid_width], dtype="object")
    for cell in table["cells"]:
        if cell["is_header"] and cell["rows"][0] <= max_header_prefix_row:
            for col in cell["cols"]:
                grid[cell["rows"][0], col] = cell["content"]
            for row in cell["rows"][1:]:
                for col in cell["cols"]:
                    grid[row, col] = ""
        else:
            grid[cell["rows"][0], cell["cols"][0]] = cell["content"]
            for col in cell["cols"][1:]:
                grid[cell["rows"][0], col] = ""
            for row in cell["rows"][1:]:
                for col in cell["cols"]:
                    grid[row, col] = ""

    header = grid[: max_header_prefix_row + 1, :]
    flattened_header = []
    for npcol in header.transpose():
        flattened_header.append(" | ".join(OrderedDict.fromkeys((c for c in npcol if c != ""))))
    df = pd.DataFrame(
        grid[max_header_prefix_row + 1 :, :],
        index=None,
        columns=flattened_header if max_header_prefix_row >= 0 else None,
    )

    return df


def tables_to_pandas(data: dict) -> list[tuple[dict, Optional[pd.DataFrame]]]:
    """
    For every table element in the provided partitioning response, create a pandas
    DataFrame representing the tabular data. Return a list containing all the elements,
    with tables paired with their corresponding DataFrames.

    Args:
        data: a response from ``partition_file``

    Example:
         .. code-block:: python

            from aryn_sdk.partition import partition_file, tables_to_pandas

            with open("my-favorite-pdf.pdf", "rb") as f:
                data = partition_file(
                    f,
                    aryn_api_key="MY-ARYN-TOKEN",
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True
                )
            elts_and_dataframes = tables_to_pandas(data)

    """
    results = []
    for e in data["elements"]:
        results.append((e, table_elem_to_dataframe(e)))

    return results
