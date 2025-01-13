from os import PathLike
from typing import BinaryIO, Literal, Optional, Union, Any
from collections.abc import Mapping
from aryn_sdk.config import ArynConfig
import requests
import sys
import json
import logging
import pandas as pd
import numpy as np
from collections import OrderedDict
from PIL import Image
import base64
import io

# URL for Aryn DocParse
ARYN_DOCPARSE_URL = "https://api.aryn.cloud/v1/document/partition"

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(sys.stderr))


class PartitionError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def partition_file(
    file: Union[BinaryIO, str, PathLike],
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    ocr_images: bool = False,
    extract_table_structure: bool = False,
    table_extraction_options: dict[str, Any] = {},
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    aps_url: Optional[str] = None,  # deprecated in favor of docparse_url
    docparse_url: Optional[str] = None,
    ssl_verify: bool = True,
    output_format: Optional[str] = None,
    output_label_options: dict[str, Any] = {},
) -> dict:
    """
    Sends file to Aryn DocParse and returns a dict of its document structure and text

    Args:
        file: (pdf, docx, doc, jpg, or png, etc.) file to partition
            (see all supported file types at https://docs.aryn.ai/docparse/formats_supported)
        aryn_api_key: Aryn api key, provided as a string
            You can get a key here: https://www.aryn.ai/get-started
        aryn_config: ArynConfig object, used for finding an api key.
            If aryn_api_key is set it will override this.
            default: The default ArynConfig looks in the env var ARYN_API_KEY and the file ~/.aryn/config.yaml
        threshold: specify the cutoff for detecting bounding boxes. Must be set to "auto" or
            a floating point value between 0.0 and 1.0.
            default: None (Aryn DocParse will choose)
        use_ocr: extract text using an OCR model instead of extracting embedded text in PDF.
            default: False
        ocr_images: attempt to use OCR to generate a text representation of detected images.
            default: False
        extract_table_structure: extract tables and their structural content.
            default: False
        table_extraction_options: Specify options for table extraction, currently only supports boolean
            'include_additional_text': if table extraction is enabled, attempt to enhance the table
            structure by merging in tokens from text extraction. This can be useful for tables with missing
            or misaligned text, and is False by default.
            default: {}
        extract_images: extract image contents in ppm format, base64 encoded.
            default: False
        selected_pages: list of individual pages (1-indexed) from the pdf to partition
            default: None
        chunking_options: Specify options for chunking the document.
            You can use the default the chunking options by setting this to {}.
            Here is an example set of chunking options:
            {
                'strategy': 'context_rich',
                'tokenizer': 'openai_tokenizer',
                'tokenizer_options': {'model_name': 'text-embedding-3-small'},
                'max_tokens': 512,
                'merge_across_pages': True
            }
            default: None
        aps_url: url of Aryn DocParse endpoint.
            Left in for backwards compatibility. Use docparse_url instead.
        docparse_url: url of Aryn DocParse endpoint.
        ssl_verify: verify ssl certificates. In databricks, set this to False to fix ssl imcompatibilities.
        output_format: controls output representation; can be set to "markdown" or "json"
            default: None (JSON elements)
        output_label_options: A dictionary for configuring output label behavior. It supports two options:
            promote_title, a boolean specifying whether to pick the largest element by font size on the first page
                from among the elements on that page that have one of the types specified in title_candidate_elements
                and promote it to type "Title" if there is no element on the first page of type "Title" already.
            title_candidate_elements, a list of strings representing the label types allowed to be promoted to
                a title.
            Here is an example set of output label options:
                {"promote_title": True, "title_candidate_elements": ["Section-header", "Caption"]}
            default: None (no element is promoted to "Title")


    Returns:
        A dictionary containing "status", "elements", and possibly "error"
        If output_format is "markdown" then it returns a dictionary of "status", "markdown", and possibly "error"

    Example:
         .. code-block:: python

            from aryn_sdk.partition import partition_file

            with open("my-favorite-pdf.pdf", "rb") as f:
                data = partition_file(
                    f,
                    aryn_api_key="MY-ARYN-API-KEY",
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True
                )
            elements = data['elements']
    """

    # If you hand me a path for the file, read it in instead of trying to send the path
    if isinstance(file, (str, PathLike)):
        with open(file, "rb") as f:
            file = io.BytesIO(f.read())

    if aryn_api_key is not None:
        if aryn_config is not None:
            _logger.warning("Both aryn_api_key and aryn_config were provided. Using aryn_api_key")
        aryn_config = ArynConfig(aryn_api_key=aryn_api_key)
    if aryn_config is None:
        aryn_config = ArynConfig()

    if aps_url is not None:
        if docparse_url is not None:
            logging.warning(
                '"aps_url" and "docparse_url" parameters were both set. "aps_url" is deprecated. Using "docparse_url".'
            )
        else:
            logging.warning('"aps_url" parameter is deprecated. Use "docparse_url" instead')
            docparse_url = aps_url
    if docparse_url is None:
        docparse_url = ARYN_DOCPARSE_URL

    options_str = _json_options(
        threshold=threshold,
        use_ocr=use_ocr,
        ocr_images=ocr_images,
        extract_table_structure=extract_table_structure,
        table_extraction_options=table_extraction_options,
        extract_images=extract_images,
        selected_pages=selected_pages,
        output_format=output_format,
        chunking_options=chunking_options,
        output_label_options=output_label_options,
    )

    _logger.debug(f"{options_str}")

    # Workaround for vcr.  See https://github.com/aryn-ai/sycamore/issues/958
    stream = True
    if "vcr" in sys.modules:
        ul3 = sys.modules.get("urllib3")
        if ul3:
            # Look for tell-tale patched method...
            mod = ul3.connectionpool.is_connection_dropped.__module__
            if "mock" in mod:
                stream = False

    files: Mapping = {"options": options_str.encode("utf-8"), "pdf": file}
    http_header = {"Authorization": "Bearer {}".format(aryn_config.api_key())}
    resp = requests.post(docparse_url, files=files, headers=http_header, stream=stream, verify=ssl_verify)

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
    if error := data.get("error"):
        code = data.get("status_code")
        if code is None:
            code = 429 if error.startswith("429: ") else 500
        if code == 429:
            prefix = "Limit exceeded"
        else:
            prefix = "Error partway through processing"
        _logger.info(f"Error from ArynPartitioner: {error}")
        raise PartitionError(f"{prefix}: {error}\nPartial Status:\n{status}", code)
    return data


def _json_options(
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    ocr_images: bool = False,
    extract_table_structure: bool = False,
    table_extraction_options: dict[str, Any] = {},
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    output_format: Optional[str] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
) -> str:
    # isn't type-checking fun
    options: dict[str, Union[float, bool, str, list[Union[list[int], int]], dict[str, Any]]] = dict()
    if threshold is not None:
        options["threshold"] = threshold
    if use_ocr:
        options["use_ocr"] = use_ocr
    if ocr_images:
        options["ocr_images"] = ocr_images
    if extract_images:
        options["extract_images"] = extract_images
    if extract_table_structure:
        options["extract_table_structure"] = extract_table_structure
    if table_extraction_options:
        options["table_extraction_options"] = table_extraction_options
    if selected_pages:
        options["selected_pages"] = selected_pages
    if output_format:
        options["output_format"] = output_format
    if chunking_options is not None:
        options["chunking_options"] = chunking_options
    if output_label_options:
        options["output_label_options"] = output_label_options

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


def convert_image_element(
    elem: dict, format: str = "PIL", b64encode: bool = False
) -> Optional[Union[Image.Image, bytes, str]]:
    """
    Convert an image element to a more useable format. If no format is specified,
    create a PIL Image object. If a format is specified, output the bytes of the image
    in that format. If b64encode is set to True, base64-encode the bytes and return them
    as a string.

    Args:
        elem: an image element from the 'elements' field of a ``partition_file`` response
        format: an optional format to output bytes of. Default is PIL
        b64encode: base64-encode the output bytes. Format must be set to use this

    Example:
         .. code-block:: python

            from aryn_sdk.partition import partition_file, convert_image_element

            with open("my-favorite-pdf.pdf", "rb") as f:
                data = partition_file(
                    f,
                    extract_images=True
                )
            image_elts = [e for e in data['elements'] if e['type'] == 'Image']

            pil_img = convert_image_element(image_elts[0])
            jpg_bytes = convert_image_element(image_elts[1], format='JPEG')
            png_str = convert_image_element(image_elts[2], format="PNG", b64encode=True)

    """
    if b64encode and format == "PIL":
        raise ValueError("b64encode was True but format was PIL. Cannot b64-encode a PIL Image")

    if elem.get("type") != "Image":
        return None

    width = elem["properties"]["image_size"][0]
    height = elem["properties"]["image_size"][1]
    mode = elem["properties"]["image_mode"]
    im = Image.frombytes(mode, (width, height), base64.b64decode(elem["binary_representation"]))

    if format == "PIL":
        return im

    buf = io.BytesIO()
    im.save(buf, format)

    if not b64encode:
        return buf.getvalue()
    else:
        return base64.b64encode(buf.getvalue()).decode("utf-8")
