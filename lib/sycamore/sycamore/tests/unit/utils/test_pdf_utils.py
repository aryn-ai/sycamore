from io import BytesIO
from pypdf import PdfReader
import pytest
import re
import sycamore
from sycamore.data import Element
from sycamore.utils.pdf_utils import (
    flatten_selected_pages,
    filter_elements_by_page,
    select_pdf_pages,
    select_pages,
)
from sycamore.tests.config import TEST_DIR


def test_flatten_selected_pages_single_page():
    result = flatten_selected_pages([3], 10)
    assert result == ([3], {1: 3})


def test_flatten_selected_pages_page_range():
    result = flatten_selected_pages([[2, 4]], 10)
    assert result == ([2, 3, 4], {1: 2, 2: 3, 3: 4})


def test_flatten_selected_pages_mixed():
    result = flatten_selected_pages([1, [3, 5], 7], 10)
    assert result == ([1, 3, 4, 5, 7], {2: 3, 3: 4, 4: 5, 5: 7})


def test_flatten_selected_pages_out_of_order():
    result = flatten_selected_pages([[5, 7], 2, [3, 4]], 10)
    assert result == ([5, 6, 7, 2, 3, 4], {1: 5, 2: 6, 3: 7, 4: 2, 5: 3, 6: 4})


def test_flatten_selected_pages_invalid_range():
    with pytest.raises(ValueError, match=re.escape("For selected_pages like [a, b] it must be that a <= b.")):
        flatten_selected_pages([[5, 3]], 10)


def test_flatten_selected_pages_overlapping():
    with pytest.raises(ValueError, match="selected_pages may not include overlapping pages."):
        flatten_selected_pages([[1, 3], [2, 4]], 10)


def test_flatten_selected_pages_out_of_bounds():
    with pytest.raises(ValueError, match="Invalid page number"):
        flatten_selected_pages([11], 10)


def test_flatten_selected_pages_zero_page():
    with pytest.raises(ValueError, match="Invalid page number"):
        flatten_selected_pages([0], 10)


def test_flatten_selected_pages_invalid_type():
    with pytest.raises(ValueError, match="Page selection must either be an integer or a 2-element list"):
        flatten_selected_pages(["1"], 10)


def test_flatten_selected_pages_empty_input():
    result = flatten_selected_pages([], 10)
    assert result == ([], {})


def test_flatten_selected_pages_all_pages():
    result = flatten_selected_pages([[1, 10]], 10)
    assert result == (list(range(1, 11)), {})


def test_flatten_selected_pages_single_page_as_range():
    result = flatten_selected_pages([[3, 3]], 10)
    assert result == ([3], {1: 3})


def test_select_pdf_pages():
    path = TEST_DIR / "resources/data/pdfs/Ray.pdf"

    bytes_out = BytesIO()
    with open(path, "rb") as infile:
        select_pdf_pages(infile, bytes_out, [1, 2, 4])

    bytes_out.seek(0)
    reader = PdfReader(bytes_out)
    assert len(reader.pages) == 3


def test_select_pdf_pages_empty():
    path = TEST_DIR / "resources/data/pdfs/Ray.pdf"

    bytes_out = BytesIO()
    with open(path, "rb") as infile:
        select_pdf_pages(infile, bytes_out, [])

    bytes_out.seek(0)
    reader = PdfReader(bytes_out)
    assert len(reader.pages) == 0


def test_select_pdf_pages_invalid_pages():
    path = TEST_DIR / "resources/data/pdfs/Ray.pdf"
    bytes_out = BytesIO()
    with pytest.raises(IndexError):
        with open(path, "rb") as infile:
            select_pdf_pages(infile, bytes_out, [1, 3, 100])


def test_filter_elements_by_page():
    elements = [
        Element(properties={"page_number": 1}),
        Element(properties={"page_number": 1}),
        Element(properties={"page_number": 2}),
        Element(properties={"page_number": 3}),
        Element(properties={"page_number": 4}),
    ]

    result = filter_elements_by_page(elements, [1])
    assert len(result) == 2 and all(e.properties["page_number"] == 1 for e in result)

    result = filter_elements_by_page(elements, [2, 4])
    assert sorted([e.properties["page_number"] for e in result]) == [1, 2]

    result = filter_elements_by_page(elements, [])
    assert len(result) == 0

    result = filter_elements_by_page(elements, [5])
    assert len(result) == 0


def test_select_pages():
    import copy

    path = TEST_DIR / "resources/data/pdfs/Ray.pdf"
    context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
    docs = context.read.binary(paths=[str(path)], binary_format="pdf").take_all()

    assert len(docs) == 1
    doc = docs[0]

    doc_fn = select_pages([[1, 2], 4])

    doc2 = copy.deepcopy(doc)
    new_doc = doc_fn(doc2)

    assert new_doc.binary_representation is not None
    assert len(new_doc.binary_representation) < len(doc.binary_representation)
    assert all(e.properties["page_number"] in [1, 2, 4] for e in new_doc.elements)
