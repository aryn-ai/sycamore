from os import PathLike
from typing import BinaryIO, Literal, Optional, Union, Any
from urllib.parse import urlparse, urlunparse
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

g_version = "0.1.17"
g_parameters = {"path_filter": "^/v1/document/partition$"}


class PartitionError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class PartitionTaskError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class PartitionTaskNotFoundError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class BoolFlag:
    """
    A boxed boolean that can be mutated and passed around by reference.
    """

    __slots__ = "val"

    def __init__(self, val: bool) -> None:
        self.val = val

    def set(self, val: bool) -> None:
        self.val = val

    def get(self) -> bool:
        return self.val


def partition_file(
    file: Union[BinaryIO, str, PathLike],
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    summarize_images: bool = False,
    ocr_language: Optional[str] = None,
    extract_table_structure: bool = False,
    text_extraction_options: Optional[dict[str, Any]] = None,
    table_extraction_options: Optional[dict[str, Any]] = None,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    aps_url: Optional[str] = None,  # deprecated in favor of docparse_url
    docparse_url: Optional[str] = None,
    ssl_verify: bool = True,
    output_format: Optional[str] = None,
    markdown_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,  # deprecated
    extra_headers: Optional[dict[str, str]] = None,
    cancel_flag: Optional[BoolFlag] = None,
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
        summarize_images: Generate a text summary of detected images using a VLM.
        ocr_language: specify the language to use for OCR. If not set, the language will be english.
            default: English
        extract_table_structure: extract tables and their structural content.
            default: False
        text_extraction_options: Specify options for text extraction, currently only supports
            'ocr_text_mode', with valid options 'vision' and 'standard'. If OCR is enabled and 'vision' specified,
            attempt to extract all non-table text using vision models, else will use the standard OCR pipeline.
            This can be useful for documents with complex layouts or non-standard fonts.
            default: {'ocr_text_mode': 'standard'}
        table_extraction_options: Specify options for table extraction. Only enabled if table extraction
            is enabled. Default is {}. Options:
            - 'include_additional_text': Attempt to enhance the table structure by merging in tokens from
                text extraction. This can be useful for tables with missing or misaligned text. Default: False
            - 'model_selection': expression to instruct DocParse how to choose which model to use for table
                extraction. See https://docs.aryn.ai/docparse/processing_options for more details. Default:
                "pixels > 500 -> deformable_detr; table_transformer"
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
        markdown_options: A dictionary for configuring markdown output behavior. It supports three options:
            include_headers, a boolean specifying whether to include headers in the markdown output, include_footers,
            a boolean specifying whether to include footers in the markdown output, and include_pagenum, a boolean
            specifying whether to include page numbers in the markdown output. Here is an example set of markdown
            options:
                {
                    "include_headers": True,
                    "include_footers": True,
                    "include_pagenum": True
                }
        output_label_options: A dictionary for configuring output label behavior. It supports three options:
            promote_title, a boolean specifying whether to pick the largest element by font size on the first page
                from among the elements on that page that have one of the types specified in title_candidate_elements
                and promote it to type "Title" if there is no element on the first page of type "Title" already.
            title_candidate_elements, a list of strings representing the label types allowed to be promoted to
                a title.
            orientation_correction, a boolean specifying whether to pagewise rotate pages to the correct orientation
                based off the orientation of text. Pages are rotated by increments of 90 degrees to correct their
                orientation.
            Here is an example set of output label options:
                {
                    "promote_title": True,
                    "title_candidate_elements": ["Section-header", "Caption"],
                    "orientation_correction": True
                }
            default: None (no element is promoted to "Title")
        trace_id: deprecated
        extra_headers: dict of HTTP headers to send to DocParse
        cancel_flag: way to interrupt partitioning from the outside


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
    return _partition_file_wrapper(
        file=file,
        aryn_api_key=aryn_api_key,
        aryn_config=aryn_config,
        threshold=threshold,
        use_ocr=use_ocr,
        summarize_images=summarize_images,
        ocr_language=ocr_language,
        extract_table_structure=extract_table_structure,
        text_extraction_options=text_extraction_options,
        table_extraction_options=table_extraction_options,
        extract_images=extract_images,
        selected_pages=selected_pages,
        chunking_options=chunking_options,
        aps_url=aps_url,
        docparse_url=docparse_url,
        ssl_verify=ssl_verify,
        output_format=output_format,
        markdown_options=markdown_options,
        output_label_options=output_label_options,
        trace_id=trace_id,
        extra_headers=extra_headers,
        cancel_flag=cancel_flag,
    )


def _partition_file_wrapper(
    file: Union[BinaryIO, str, PathLike],
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    summarize_images: bool = False,
    ocr_language: Optional[str] = None,
    extract_table_structure: bool = False,
    text_extraction_options: Optional[dict[str, Any]] = None,
    table_extraction_options: Optional[dict[str, Any]] = None,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    aps_url: Optional[str] = None,  # deprecated in favor of docparse_url
    docparse_url: Optional[str] = None,
    ssl_verify: bool = True,
    output_format: Optional[str] = None,
    markdown_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
    webhook_url: Optional[str] = None,
    trace_id: Optional[str] = None,  # deprecated
    extra_headers: Optional[dict[str, str]] = None,
    cancel_flag: Optional[BoolFlag] = None,
):
    """Do not call this function directly. Use partition_file or partition_file_async_submit instead."""

    # If you hand me a path for the file, read it in instead of trying to send the path
    should_close = False
    try:
        if isinstance(file, (str, PathLike)):
            file = open(file, "rb")
            should_close = True
        return _partition_file_inner(
            file=file,
            aryn_api_key=aryn_api_key,
            aryn_config=aryn_config,
            threshold=threshold,
            use_ocr=use_ocr,
            summarize_images=summarize_images,
            ocr_language=ocr_language,
            extract_table_structure=extract_table_structure,
            text_extraction_options=text_extraction_options,
            table_extraction_options=table_extraction_options,
            extract_images=extract_images,
            selected_pages=selected_pages,
            chunking_options=chunking_options,
            aps_url=aps_url,
            docparse_url=docparse_url,
            ssl_verify=ssl_verify,
            output_format=output_format,
            markdown_options=markdown_options,
            output_label_options=output_label_options,
            trace_id=trace_id,
            extra_headers=extra_headers,
            cancel_flag=cancel_flag,
            webhook_url=webhook_url,
        )
    finally:
        if should_close and isinstance(file, BinaryIO):
            file.close()


def _partition_file_inner(
    file: BinaryIO,
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    summarize_images: bool = False,
    ocr_language: Optional[str] = None,
    extract_table_structure: bool = False,
    text_extraction_options: Optional[dict[str, Any]] = None,
    table_extraction_options: Optional[dict[str, Any]] = None,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    aps_url: Optional[str] = None,  # deprecated in favor of docparse_url
    docparse_url: Optional[str] = None,
    ssl_verify: bool = True,
    output_format: Optional[str] = None,
    markdown_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,  # deprecated
    extra_headers: Optional[dict[str, str]] = None,
    cancel_flag: Optional[BoolFlag] = None,
    webhook_url: Optional[str] = None,
):
    """Do not call this function directly. Use partition_file or partition_file_async_submit instead."""

    aryn_config = _process_config(aryn_api_key, aryn_config)

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
    source = extra_headers.pop("X-Aryn-Origin", "aryn-sdk") if extra_headers else "aryn-sdk"

    options_str = _json_options(
        threshold=threshold,
        use_ocr=use_ocr,
        summarize_images=summarize_images,
        ocr_language=ocr_language,
        extract_table_structure=extract_table_structure,
        text_extraction_options=text_extraction_options,
        table_extraction_options=table_extraction_options,
        extract_images=extract_images,
        selected_pages=selected_pages,
        output_format=output_format,
        chunking_options=chunking_options,
        markdown_options=markdown_options,
        output_label_options=output_label_options,
        source=source,
    )

    _logger.debug(f"{options_str}")

    files: Mapping = {"options": options_str.encode("utf-8"), "pdf": file}
    headers = _generate_headers(aryn_config.api_key(), webhook_url, trace_id, extra_headers)
    resp = requests.post(docparse_url, files=files, headers=headers, stream=_should_stream(), verify=ssl_verify)

    raise_error_on_non_2xx(resp)

    content = []
    partial_line = []
    in_bulk = False
    for part in resp.iter_content(None):
        # A big doc could take a while; we may be asked to bail out early
        if cancel_flag and cancel_flag.get():
            resp.close()
            break

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


def raise_error_on_non_2xx(resp: requests.Response) -> None:
    if resp.status_code < 200 or resp.status_code > 299:
        raise requests.exceptions.HTTPError(
            f"Error: status_code: {resp.status_code}, reason: {resp.text}", response=resp
        )


def _process_config(aryn_api_key: Optional[str] = None, aryn_config: Optional[ArynConfig] = None) -> ArynConfig:
    if aryn_api_key is not None:
        if aryn_config is not None:
            _logger.warning("Both aryn_api_key and aryn_config were provided. Using aryn_api_key")
        aryn_config = ArynConfig(aryn_api_key=aryn_api_key)
    if aryn_config is None:
        aryn_config = ArynConfig()
    return aryn_config


def _generate_headers(
    aryn_api_key: str,
    webhook_url: Optional[str] = None,
    trace_id: Optional[str] = None,  # deprecated
    extra_headers: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    # Start with user-supplied headers so they can't stomp official ones.
    headers = extra_headers.copy() if extra_headers else {}
    if webhook_url:
        headers["X-Aryn-Webhook"] = webhook_url
    if trace_id:
        headers["X-Aryn-Trace-ID"] = trace_id  # deprecated
    headers["Authorization"] = f"Bearer {aryn_api_key}"
    headers["User-Agent"] = f"aryn-sdk/{g_version}"
    return headers


def _should_stream() -> bool:
    # Workaround for vcr.  See https://github.com/aryn-ai/sycamore/issues/958
    stream = True
    if "vcr" in sys.modules:
        ul3 = sys.modules.get("urllib3")
        if ul3:
            # Look for tell-tale patched method...
            mod = ul3.connectionpool.is_connection_dropped.__module__
            if "mock" in mod:
                stream = False
    return stream


def _json_options(
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    summarize_images: bool = False,
    ocr_language: Optional[str] = None,
    extract_table_structure: bool = False,
    text_extraction_options: Optional[dict[str, Any]] = None,
    table_extraction_options: Optional[dict[str, Any]] = None,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    output_format: Optional[str] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    markdown_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
    source: str = "aryn-sdk",
) -> str:
    # isn't type-checking fun
    options: dict[str, Union[float, bool, str, list[Union[list[int], int]], dict[str, Any]]] = dict()
    if threshold is not None:
        options["threshold"] = threshold
    if use_ocr:
        options["use_ocr"] = use_ocr
    if summarize_images:
        options["summarize_images"] = summarize_images
    if ocr_language:
        options["ocr_language"] = ocr_language
    if extract_images:
        options["extract_images"] = extract_images
    if extract_table_structure:
        options["extract_table_structure"] = extract_table_structure
    if text_extraction_options:
        options["text_extraction_options"] = text_extraction_options
    if table_extraction_options:
        options["table_extraction_options"] = table_extraction_options
    if selected_pages:
        options["selected_pages"] = selected_pages
    if output_format:
        options["output_format"] = output_format
    if chunking_options is not None:
        options["chunking_options"] = chunking_options
    if markdown_options:
        options["markdown_options"] = markdown_options
    if output_label_options:
        options["output_label_options"] = output_label_options

    options["source"] = source

    return json.dumps(options)


def partition_file_async_submit(
    file: Union[BinaryIO, str, PathLike],
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    threshold: Optional[Union[float, Literal["auto"]]] = None,
    use_ocr: bool = False,
    summarize_images: bool = False,
    ocr_language: Optional[str] = None,
    extract_table_structure: bool = False,
    text_extraction_options: Optional[dict[str, Any]] = None,
    table_extraction_options: Optional[dict[str, Any]] = None,
    extract_images: bool = False,
    selected_pages: Optional[list[Union[list[int], int]]] = None,
    chunking_options: Optional[dict[str, Any]] = None,
    aps_url: Optional[str] = None,  # deprecated in favor of docparse_url
    docparse_url: Optional[str] = None,
    ssl_verify: bool = True,
    output_format: Optional[str] = None,
    markdown_options: Optional[dict[str, Any]] = None,
    output_label_options: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,  # deprecated
    extra_headers: Optional[dict[str, str]] = None,
    webhook_url: Optional[str] = None,
    async_submit_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Submits a file to be partitioned asynchronously. Meant to be used in tandem with `partition_file_async_result`.

    `partition_file_async_submit` takes the same arguments as `partition_file`, and in addition it accepts a str
    `webhook_url` argument which is a URL Aryn will send a POST request to when the task stops and an str
    `async_submit_url` argument that can be used to override where the task is submitted to.

    Set the `docparse_url` argument to the url of the synchronous endpoint, and this function will automatically
    change it to the async endpoint as long as `async_submit_url` is not set.

    For examples of usage see README.md

    Args:
        Includes All Arguments `partition_file` accepts plus those below:
        ...
        webhook_url: A URL to send a POST request to when the task is done. The resulting POST request will have a
            body like: {"done": [{"task_id": "aryn:j-47gpd3604e5tz79z1jro5fc"}]}
        async_submit_url: When set, this will override the endpoint the task is submitted to.

    Returns:
        A dictionary containing the key "task_id" the value of which can be used with the `partition_file_async_result`
        function to get the results and check the status of the async task.
    """

    if async_submit_url:
        docparse_url = async_submit_url
    elif not aps_url and not docparse_url:
        docparse_url = _convert_sync_to_async_url(ARYN_DOCPARSE_URL, "/submit", truncate=False)
    else:
        if aps_url:
            aps_url = _convert_sync_to_async_url(aps_url, "/submit", truncate=False)
        if docparse_url:
            docparse_url = _convert_sync_to_async_url(docparse_url, "/submit", truncate=False)

    return _partition_file_wrapper(
        file=file,
        aryn_api_key=aryn_api_key,
        aryn_config=aryn_config,
        threshold=threshold,
        use_ocr=use_ocr,
        summarize_images=summarize_images,
        ocr_language=ocr_language,
        extract_table_structure=extract_table_structure,
        text_extraction_options=text_extraction_options,
        table_extraction_options=table_extraction_options,
        extract_images=extract_images,
        selected_pages=selected_pages,
        chunking_options=chunking_options,
        aps_url=aps_url,
        docparse_url=docparse_url,
        ssl_verify=ssl_verify,
        output_format=output_format,
        markdown_options=markdown_options,
        output_label_options=output_label_options,
        trace_id=trace_id,
        extra_headers=extra_headers,
        webhook_url=webhook_url,
    )


def _convert_sync_to_async_url(url: str, prefix: str, *, truncate: bool) -> str:
    parsed_url = urlparse(url)
    assert parsed_url.path.startswith("/v1/")
    if parsed_url.path.startswith("/v1/async/submit"):
        return url
    ary = list(parsed_url)
    if truncate:
        ary[2] = f"/v1/async{prefix}"  # path
    else:
        ary[2] = f"/v1/async{prefix}{parsed_url.path[3:]}"  # path
    return urlunparse(ary)


def partition_file_async_result(
    task_id: str,
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    ssl_verify: bool = True,
    async_result_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get the results of an asynchronous partitioning task by task_id. Meant to be used with
    `partition_file_async_submit`.

    For examples of usage see README.md

    Raises a `PartitionTaskNotFoundError` if the not task with the task_id can be found.

    Returns:
        A dict containing "status" and "status_code". When "status" is "done", the returned dict also contains "result"
        which contains what would have been returned had `partition_file` been called directly. "status" can be "done"
        or "pending".

        Unlike `partition_file`, this function does not raise an Exception if the partitioning failed.
    """
    if not async_result_url:
        async_result_url = _convert_sync_to_async_url(ARYN_DOCPARSE_URL, "/result", truncate=True)

    aryn_config = _process_config(aryn_api_key, aryn_config)

    specific_task_url = f"{async_result_url.rstrip('/')}/{task_id}"
    headers = _generate_headers(aryn_config.api_key())
    response = requests.get(
        specific_task_url, params=g_parameters, headers=headers, stream=_should_stream(), verify=ssl_verify
    )
    if response.status_code == 200:
        return {"task_status": "done", "result": response.json()}
    elif response.status_code == 202:
        return {"task_status": "pending"}
    elif response.status_code == 404:
        raise PartitionTaskNotFoundError("No such task", response.status_code)
    else:
        raise_error_on_non_2xx(response)
        raise PartitionTaskError("Unexpected response code", response.status_code)


def partition_file_async_cancel(
    task_id: str,
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    ssl_verify: bool = True,
    async_cancel_url: Optional[str] = None,
) -> None:
    """
    Cancel an asynchronous partitioning task by task_id. Meant to be used with `partition_file_async_submit`.

    Raises an exception if there is no cancellable task found with the given task_id. A task can only be successfully
    cancelled once.

    For an example of usage see README.md
    """
    if not async_cancel_url:
        async_cancel_url = _convert_sync_to_async_url(ARYN_DOCPARSE_URL, "/cancel", truncate=True)

    aryn_config = _process_config(aryn_api_key, aryn_config)

    specific_task_url = f"{async_cancel_url.rstrip('/')}/{task_id}"
    headers = _generate_headers(aryn_config.api_key())
    response = requests.post(
        specific_task_url, params=g_parameters, headers=headers, stream=_should_stream(), verify=ssl_verify
    )
    if response.status_code == 200:
        return
    elif response.status_code == 404:
        raise PartitionTaskNotFoundError("No such task", response.status_code)
    else:
        raise_error_on_non_2xx(response)
        raise PartitionTaskError("Unexpected response code.", response.status_code)


def partition_file_async_list(
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    ssl_verify: bool = True,
    async_list_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    List pending async tasks. For an example of usage see README.md

    Returns:
        A dict like the one below which maps task_ids to a dict containing details of the respective task.

        {
            "aryn:j-sc0v0lglkauo774pioflp4l": {
                "state": "run"
            },
            "aryn:j-b9xp7ny0eejvqvbazjhg8rn": {
                "state": "run"
            }
        }
    """
    if not async_list_url:
        async_list_url = _convert_sync_to_async_url(ARYN_DOCPARSE_URL, "/list", truncate=True)

    aryn_config = _process_config(aryn_api_key, aryn_config)

    headers = _generate_headers(aryn_config.api_key())
    response = requests.get(
        async_list_url, params=g_parameters, headers=headers, stream=_should_stream(), verify=ssl_verify
    )
    raise_error_on_non_2xx(response)
    if response.status_code != 200:
        raise PartitionTaskError("Unexpected response code", response.status_code)

    result = response.json()

    tasks = result["tasks"]
    for v in tasks.values():
        v.pop("action", None)
    return tasks


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
                    aryn_api_key="MY-ARYN-API-KEY",
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
