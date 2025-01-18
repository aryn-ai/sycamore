from os import PathLike
from typing import Any, BinaryIO, Literal, Optional, Union
from aryn_sdk.partition.partition import convert_image_element, tables_to_pandas, ARYN_DOCPARSE_URL
import pytest
import json
import time
from pathlib import Path
import inspect

from aryn_sdk.partition import (
    partition_file,
    partition_file_submit_async,
    partition_file_result_async,
    NoSuchAsyncPartitionerJobError,
    PartitionError,
)
from aryn_sdk.config import ArynConfig
from requests.exceptions import HTTPError

RESOURCE_DIR = Path(__file__).parent / "resources"


# Unit tests
@pytest.mark.parametrize(
    "pdf, kwargs, response",
    [
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {},
            RESOURCE_DIR / "json" / "3m_output.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {"use_ocr": True},
            RESOURCE_DIR / "json" / "3m_output_ocr.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {"use_ocr": True, "extract_table_structure": True},
            RESOURCE_DIR / "json" / "3m_output_ocr_table.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {"extract_table_structure": True},
            RESOURCE_DIR / "json" / "3m_output_table.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "SPsort.pdf",
            {},
            RESOURCE_DIR / "json" / "SPsort_output.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "SPsort.pdf",
            {"extract_images": True},
            RESOURCE_DIR / "json" / "SPsort_output_images.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "SPsort.pdf",
            {"extract_images": True, "selected_pages": [0]},
            RESOURCE_DIR / "json" / "SPsort_output_images_page0.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "SPsort.pdf",
            {"extract_images": True, "selected_pages": [1]},
            RESOURCE_DIR / "json" / "SPsort_output_images_page1.json",
        ),
    ],
)
def test_partition(pdf, kwargs, response, mocker):

    with open(response, "rb") as f:
        byteses = f.read()

    response_data = json.loads(byteses.decode("utf-8"))
    resp_object = mocker.Mock()
    resp_object.status_code = 200

    # Mock the response from the file,
    resp_object.iter_content.return_value = byteses.split(sep=b"\n")

    mocker.patch("requests.post").return_value = resp_object

    with open(pdf, "rb") as f:
        if kwargs.get("selected_pages") == [0]:
            with pytest.raises(PartitionError) as einfo:
                new_response = partition_file(f, **kwargs)
            assert "Invalid page number (0)" in str(einfo.value)
        else:
            new_response = partition_file(f, **kwargs)
            assert new_response == response_data


# Integration tests


@pytest.mark.parametrize(
    "pdf, kwargs, response",
    [
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {},
            RESOURCE_DIR / "json" / "3m_output.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "3m_table.pdf",
            {"use_ocr": True, "extract_table_structure": True},
            RESOURCE_DIR / "json" / "3m_output_ocr_table.json",
        ),
        (
            RESOURCE_DIR / "pdfs" / "SPsort.pdf",
            {"extract_images": True, "selected_pages": [1]},
            RESOURCE_DIR / "json" / "SPsort_output_images_page1.json",
        ),
    ],
)
def test_partition_it(pdf, kwargs, response):

    with open(response, "r") as f:
        response_data = json.load(f)

    with open(pdf, "rb") as f:
        new_response = partition_file(f, **kwargs)

    assert response_data["elements"] == new_response["elements"]


def test_partition_it_zero_page():

    with pytest.raises(PartitionError) as einfo:
        with open(RESOURCE_DIR / "pdfs" / "SPsort.pdf", "rb") as f:
            partition_file(f, selected_pages=[0])

    assert "Invalid page number (0)" in str(einfo.value)


def test_partition_it_no_api_key():
    with pytest.raises(HTTPError) as einfo:
        with open(RESOURCE_DIR / "pdfs" / "3m_table.pdf", "rb") as f:
            partition_file(f, aryn_api_key="")
    assert einfo.value.response.status_code == 403
    assert einfo.value.response.json().get("detail") == "Not authenticated"


def test_data_to_pandas():
    with open(RESOURCE_DIR / "json" / "3m_output_ocr_table.json", "r") as f:
        data = json.load(f)
    elts_and_dfs = tables_to_pandas(data)
    assert len(elts_and_dfs) == 5
    df = elts_and_dfs[2][1]
    assert df is not None
    assert df.columns.to_list() == ["(Millions)", "2018", "2017", "2016"]
    assert df["2018"][13] == "134"


def test_convert_img():
    with open(RESOURCE_DIR / "image" / "partitioning_output.json", "r") as f:
        data = json.load(f)
    ims = [e for e in data["elements"] if e["type"] == "Image"]

    jpg_bytes = convert_image_element(ims[0], format="JPEG")
    with open(RESOURCE_DIR / "image" / "jpeg_bytes.jpeg", "rb") as f:
        real_byteses = f.read().strip()
    assert jpg_bytes == real_byteses

    png_str = convert_image_element(ims[0], format="PNG", b64encode=True)
    with open(RESOURCE_DIR / "image" / "pngb64str.txt", "r") as f:
        real_str = f.read().strip()
    assert png_str == real_str


def test_invalid_job_id():
    with pytest.raises(NoSuchAsyncPartitionerJobError):
        partition_file_result_async("INVALID_JOB_ID")


def test_partition_file_submit_async(mocker):
    data = b'{"job_id": "1234"}'
    expected_response = json.loads(data.decode("utf-8"))

    mocked_response = mocker.Mock()
    mocked_response.status_code = 202
    mocked_response.iter_content.return_value = data.split(sep=b"\n")

    mocker.patch("requests.post").return_value = mocked_response

    with open(RESOURCE_DIR / "pdfs" / "3m_table.pdf", "rb") as f:
        response = partition_file_submit_async(f)

    assert response == expected_response


def test_partition_file_async():
    with open(RESOURCE_DIR / "pdfs" / "3m_table.pdf", "rb") as f:
        job_id = partition_file_submit_async(f)["job_id"]

    start = time.time()
    actual_result = None
    while not actual_result and time.time() - start < 60 * 5:
        actual_result = partition_file_result_async(job_id)
        time.sleep(5)

    with open(RESOURCE_DIR / "json" / "3m_output.json", "rb") as f:
        expected_result = json.load(f)

    assert expected_result["elements"] == actual_result["elements"]


def test_partiton_file_async_url_forwarding(mocker):
    def call_partition_file(base_url: str):
        partition_file_submit_async("", docparse_url=base_url)
        partition_file_submit_async("", aps_url=base_url)
        partition_file_submit_async(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, base_url)
        partition_file_submit_async(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "wrong", base_url)
        partition_file_submit_async(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "wrong", docparse_url=base_url)
        partition_file_submit_async("", aps_url=base_url, docparse_url=base_url)

    standard_async_url = ARYN_DOCPARSE_URL.replace("/v1/", "/v1/async/submit/")

    def check_standard_url(
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
    ):
        url = docparse_url or aps_url
        assert url == standard_async_url

    mocker.patch("inspect.getfullargspec").return_value = inspect.getfullargspec(partition_file)
    mocker.patch("aryn_sdk.partition.partition.partition_file", side_effect=check_standard_url)
    partition_file_submit_async("")
    call_partition_file(ARYN_DOCPARSE_URL)
    call_partition_file(standard_async_url)

    nonstandard_url_example = "http://localhost:8000/v1/document/partition"
    nonstandard_async_url_example = nonstandard_url_example.replace("/v1/", "/v1/async/submit/")

    def check_nonstandard_url(
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
    ):
        url = docparse_url or aps_url
        assert url == nonstandard_async_url_example

    mocker.patch("aryn_sdk.partition.partition.partition_file", side_effect=check_nonstandard_url)
    call_partition_file(nonstandard_url_example)
    call_partition_file(nonstandard_async_url_example)
