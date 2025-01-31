from os import PathLike
from typing import BinaryIO, Union
from aryn_sdk.partition.partition import convert_image_element, tables_to_pandas, ARYN_DOCPARSE_URL
import pytest
import json
import time
from pathlib import Path
import logging

from aryn_sdk.partition import (
    partition_file,
    partition_file_async_submit,
    partition_file_async_result,
    partition_file_async_cancel,
    partition_file_async_list,
    PartitionError,
)
from requests.exceptions import HTTPError

RESOURCE_DIR = Path(__file__).parent / "resources"
ASYNC_TIMEOUT = 60 * 5  # 5 minutes in seconds


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


def test_partition_with_unsupported_file_format():
    with pytest.raises(PartitionError):
        with open(RESOURCE_DIR / "image" / "unsupported-format-test-document-image.heic", "rb") as f:
            partition_file(f)


def test_partition_it_zero_page():
    with pytest.raises(PartitionError) as einfo:
        with open(RESOURCE_DIR / "pdfs" / "SPsort.pdf", "rb") as f:
            partition_file(f, selected_pages=[0])
    assert "selected_pages must not have empty or zero terms" in str(einfo.value)


def test_partition_it_no_api_key():
    with pytest.raises(HTTPError) as einfo:
        with open(RESOURCE_DIR / "pdfs" / "3m_table.pdf", "rb") as f:
            partition_file(f, aryn_api_key="")
    assert einfo.value.response.status_code == 403
    assert einfo.value.response.json().get("detail") == "Not authenticated"


def test_partition_file_auto_rotation():
    expected = json.loads(open(RESOURCE_DIR / "json" / "rotated.json", "r").read())
    actual = partition_file(
        RESOURCE_DIR / "pdfs" / "rotated.pdf", output_label_options={"orientation_correction": True}
    )
    assert actual["elements"] == expected["elements"]


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
    response = partition_file_async_result("INVALID_JOB_ID")
    assert response["status"] == "no_such_job"


def test_partition_file_async_submit(mocker):
    data = b'{"job_id": "1234"}'
    expected_response = json.loads(data.decode())

    mocked_response = mocker.Mock()
    mocked_response.status_code = 202
    mocked_response.iter_content.return_value = (data,)

    mocker.patch("requests.post").return_value = mocked_response

    with open(RESOURCE_DIR / "pdfs" / "3m_table.pdf", "rb") as f:
        response = partition_file_async_submit(f)

    assert response == expected_response


def test_partiton_file_async_url_forwarding(mocker):
    def call_partition_file(base_url: str):
        partition_file_async_submit("", docparse_url=base_url)
        partition_file_async_submit("", aps_url=base_url)
        partition_file_async_submit("", aps_url="https://example.com/v1/document/partition", docparse_url=base_url)
        partition_file_async_submit("", aps_url=base_url, docparse_url=base_url)

    standard_async_url = ARYN_DOCPARSE_URL.replace("/v1/", "/v1/async/submit/")

    def check_standard_url(
        file: Union[BinaryIO, str, PathLike],
        **kwargs,
    ) -> None:
        url = kwargs.get("docparse_url") or kwargs.get("aps_url")
        assert url == standard_async_url

    mocker.patch("aryn_sdk.partition.partition._partition_file_inner", side_effect=check_standard_url)
    partition_file_async_submit("")
    call_partition_file(ARYN_DOCPARSE_URL)
    call_partition_file(standard_async_url)

    nonstandard_url_example = "http://localhost:8000/v1/document/partition"
    nonstandard_async_url_example = nonstandard_url_example.replace("/v1/", "/v1/async/submit/")

    def check_nonstandard_url(
        file: Union[BinaryIO, str, PathLike],
        **kwargs,
    ) -> None:
        url = kwargs.get("docparse_url") or kwargs.get("aps_url")
        assert url == nonstandard_async_url_example

    mocker.patch("aryn_sdk.partition.partition._partition_file_inner", side_effect=check_nonstandard_url)
    call_partition_file(nonstandard_url_example)
    call_partition_file(nonstandard_async_url_example)


def test_partition_file_async_with_unsupported_file_format():
    with open(RESOURCE_DIR / "image" / "unsupported-format-test-document-image.heic", "rb") as f:
        job_id = partition_file_async_submit(f)["job_id"]

    start = time.time()
    while True:
        actual_result = partition_file_async_result(job_id)
        if actual_result["status"] != "pending" or time.time() - start >= ASYNC_TIMEOUT:
            break
        time.sleep(1)
    assert actual_result["status"] == "done"
    assert actual_result["result"] is not None
    assert actual_result["result"]["status_code"] == 500
    assert actual_result["result"]["error"] == "500: Failed to convert file to pdf"


def test_multiple_partition_file_async():
    num_jobs = 4
    job_ids = []

    before = partition_file_async_list()
    logging.info(f"List before:\n{json.dumps(before, indent=4)}")

    for i in range(num_jobs):
        logging.info(f"Submitting job {i + 1}/{num_jobs}")
        job_id = partition_file_async_submit(RESOURCE_DIR / "pdfs" / "FR-2002-05-03-TRUNCATED-40.pdf")["job_id"]
        logging.info(f"\tJob ID: {job_id}")
        job_ids.append(job_id)

    after = partition_file_async_list()
    logging.info(f"List after:\n{json.dumps(after, indent=4)}")
    assert len(after) >= num_jobs
    for job_id in job_ids:
        assert job_id in after

    for i, job_id in enumerate(job_ids):
        logging.info(f"Polling job ({job_id}) {i + 1}/{num_jobs}")
        start = time.time()
        while True:
            actual_result = partition_file_async_result(job_id)
            if actual_result["status"] != "pending" or time.time() - start >= ASYNC_TIMEOUT:
                break
            time.sleep(1)
            logging.info(f"\tContinuing to Poll Job {job_id} ({i + 1}/{num_jobs})")
        assert actual_result["status"] == "done"
        assert len(actual_result["result"]["elements"]) > 1000


def test_partition_file_async_cancel():
    with open(RESOURCE_DIR / "pdfs" / "FR-2002-05-03-TRUNCATED-40.pdf", "rb") as f:
        job_id = partition_file_async_submit(f)["job_id"]

    before_cancel_result = partition_file_async_result(job_id)
    assert before_cancel_result["status"] == "pending"
    assert partition_file_async_cancel(job_id)

    # Cancellation is not reflected in the result immediately
    for _ in range(10):
        time.sleep(0.1)
        after_cancel_result = partition_file_async_result(job_id)
        if after_cancel_result["status"] != "pending":
            break
        assert after_cancel_result["status"] == "pending"
    assert after_cancel_result["status"] == "no_such_job"


def test_smoke_webhook(mocker):
    data = b'{"job_id": "1234"}'

    webhook_url = "TEST"

    mocked_response = mocker.Mock()
    mocked_response.status_code = 202
    mocked_response.iter_content.return_value = data.split(sep=b"\n")

    def check_webhook(*args, headers, **kwargs):
        assert "X-Aryn-Webhook" in headers
        assert headers.get("X-Aryn-Webhook") == webhook_url
        return mocked_response

    fake_post = mocker.patch("requests.post", side_effect=check_webhook)
    partition_file_async_submit(RESOURCE_DIR / "pdfs" / "3m_table.pdf", webhook_url=webhook_url)
    fake_post.assert_called()
