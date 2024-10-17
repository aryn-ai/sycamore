from io import BytesIO
import logging
from typing import BinaryIO, Callable, Union
from PIL import Image

from pypdf import PdfReader, PdfWriter

from sycamore import DocSet
from sycamore.functions.document import DrawBoxes, split_and_convert_to_image
from sycamore.utils.image_utils import show_images
from sycamore.data import Document, Element
import json

logger = logging.getLogger(__name__)


def show_pages(docset: DocSet, limit: int = 2):
    documents = (
        docset.flat_map(split_and_convert_to_image)
        .limit(limit)
        .map_batch(DrawBoxes(), f_constructor_kwargs={"draw_table_cells": True})
        .take(limit)
    )
    images = [
        Image.open(BytesIO(doc.binary_representation)) for doc in documents if doc.binary_representation is not None
    ]

    show_images(images)


def flatten_selected_pages(
    selected_pages: list[Union[int, list[int]]], page_count: int
) -> tuple[list[int], dict[int, int]]:
    """
    Accepts a page selection that consists of a page (like [11] ), a page range (like [[25,30]] ),
    or a combination of both (like [11, [25,30]] ). Pages are 1-indexed.

    Returns a list of individual page numbers and a dictionary that maps the new page numbers to the
    original page numbers in cases where the two are not equal.
    """

    page_list = []
    present_pages = set()
    remapped_pages = {}
    new_page = 1
    for selection in selected_pages:
        if isinstance(selection, int):
            selection = [selection, selection]
        if isinstance(selection, list):
            subset_start, subset_end = selection
            if subset_end < subset_start:
                raise ValueError("For selected_pages like [a, b] it must be that a <= b.")
            for page_num in range(subset_start, subset_end + 1):
                if page_num in present_pages:
                    raise ValueError("selected_pages may not include overlapping pages.")
                if page_num <= 0 or page_num > page_count:
                    raise ValueError(
                        f"Invalid page number ({page_num}): for this document,"
                        f"page numbers must be at least 1 and at most {page_count}"
                    )
                present_pages.add(page_num)
                page_list.append(page_num)

                if page_num != new_page:
                    remapped_pages[new_page] = page_num

                new_page = new_page + 1

        else:
            raise ValueError("Page selection must either be an integer or a 2-element list [integer, integer]")
    return (page_list, remapped_pages)


def select_pdf_pages(input: BinaryIO, out: BinaryIO, page_list: list[int]) -> None:
    input.seek(0)
    with PdfReader(input) as pdf_reader, PdfWriter() as pdf_writer:
        for page_num in page_list:
            pdf_writer.add_page(pdf_reader.pages[page_num - 1])
        pdf_writer.write_stream(out)  # see pypdf issue #2905
    out.flush()


def filter_elements_by_page(elements: list[Element], page_numbers: list[int]) -> list[Element]:
    page_map = {num: idx + 1 for idx, num in enumerate(page_numbers)}
    new_elements = []
    for element in elements:
        page_number = element.properties.get("page_number")
        if page_number is not None and page_number in page_map:
            # renumber pages so the elements reference the pages in the new document.
            element.properties["page_number"] = page_map[page_number]
            new_elements.append(element)
    return new_elements


def select_pages(page_selection: list[Union[int, list[int]]]) -> Callable[[Document], Document]:
    """
    Returns a function that selects pages from a PDF document based on a list of page selections.
    Each selection can be a single page number or a range of page numbers. Page numbers are 1-indexed.

    Examples:
       [1,2,3] pages 1, 2, and 3
       [[1,3], 5] pages 1, 2, 3, and 5
       [[1,3], [5,7] pages 1, 2, 3, and 5, 6, 7
       [2, 1, [4, 6]] pages 2, 1, 4, 5, 6, in that order

    Args:
       page_selection: A list of page numbers or page ranges to select. Page numbers are 1-indexed.

    """

    def select_pages_fn(doc: Document) -> Document:
        if doc.binary_representation is None:
            logging.warning("No binary_representation found in doc {doc.doc_id}. Skipping page selection.")
            return doc

        with PdfReader(BytesIO(doc.binary_representation)) as reader:
            page_count = len(reader.pages)

        page_list, remapped_pages = flatten_selected_pages(page_selection, page_count)

        outstream = BytesIO()
        select_pdf_pages(BytesIO(doc.binary_representation), outstream, page_list=page_list)
        doc.binary_representation = outstream.getvalue()

        doc.properties["remapped_pages"] = remapped_pages
        new_elements = filter_elements_by_page(doc.elements, page_list)
        doc.elements = new_elements
        return doc

    return select_pages_fn


def enumerate_images_and_tables(m_pages: list[Document]):
    from IPython.display import display, HTML

    num_pages = len(m_pages)
    for i in range(0, num_pages):
        m_page = m_pages[i]
        print("Path: ", m_page.properties["path"])
        for e in m_page.elements:
            if e.type == "Image":
                print("Image summary: ", e.properties["summary"], "\n")
                print()
            if e.type == "table":
                table_text_html = e.get("table")
                if table_text_html:
                    display(HTML(table_text_html.to_html()))
                print()


def display_page_and_table_properties(some_pages: list[Document]):
    from IPython.display import display, HTML

    for m_page in some_pages:
        print("Page props: ")
        display(m_page.properties["entity"])
        print()
        for e in m_page.elements:
            if e and e.type == "table":
                print("Element Type: ", e.type)
                print("Element Properties: ", json.dumps(e.properties, indent=2, default=str))
                display(HTML(e.text_representation))
