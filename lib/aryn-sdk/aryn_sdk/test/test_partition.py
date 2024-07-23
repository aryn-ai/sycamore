from aryn_sdk.partition.partition import tables_to_pandas
import pytest
import json
from pathlib import Path

from aryn_sdk.partition import partition_file
from requests.exceptions import HTTPError

RESOURCE_DIR = Path(__file__).parent / "resources"


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
    with open(response, "r") as f:
        response_data = json.load(f)
    resp_object = mocker.Mock()
    resp_object.status_code = 200
    resp_object.json.return_value = response_data

    mocker.patch("requests.post").return_value = resp_object

    with open(pdf, "rb") as f:
        new_response = partition_file(f, **kwargs)
    assert new_response == response_data


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
    df = elts_and_dfs[0][1]
    assert df is not None
    assert df.columns.to_list() == ["(Millions)", "2018", "2017", "2016"]
    assert df["2018"][13] == "134"
