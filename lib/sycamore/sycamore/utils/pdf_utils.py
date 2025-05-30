from io import BytesIO
from contextlib import nullcontext
import logging
from typing import Any, BinaryIO, Callable, cast, Union
from PIL import Image

from pypdf import PdfReader, PdfWriter
import pdf2image

from sycamore import DocSet
from sycamore.functions.document import DrawBoxes, split_and_convert_to_image
from sycamore.utils.image_utils import show_images, crop_to_bbox
from sycamore.data import Document, Element, ImageElement
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

                if page_num != len(page_list):
                    remapped_pages[len(page_list)] = page_num

        else:
            raise ValueError("Page selection must either be an integer or a 2-element list [integer, integer]")
    return (page_list, remapped_pages)


def select_pdf_pages(input: Union[BinaryIO, PdfReader], out: BinaryIO, page_list: list[int]) -> None:
    if isinstance(input, PdfReader):
        read_cm: Any = nullcontext(input)  # Caller is responsible for cleaning up.
    else:
        input.seek(0)
        read_cm = PdfReader(input)

    with read_cm as pdf_reader, PdfWriter() as pdf_writer:
        for page_num in page_list:
            pdf_writer.add_page(pdf_reader.pages[page_num - 1])
        pdf_writer.write_stream(out)  # see pypdf issue #2905
    out.flush()


def filter_elements_by_page(elements: list[Element], page_numbers: list[int]) -> list[Element]:
    page_map = {num: idx + 1 for idx, num in enumerate(page_numbers)}
    new_elements = []
    for element in elements:
        page_number = element.properties.get("page_number")
        if (new_number := page_map.get(cast(int, page_number))) is not None:
            # renumber pages so the elements reference the pages in the new document.
            element.properties["page_number"] = new_number
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
            logging.warning(f"No binary_representation found in doc {doc.doc_id}. Skipping page selection.")
            return doc

        outstream = BytesIO()

        with PdfReader(BytesIO(doc.binary_representation)) as reader:
            page_count = len(reader.pages)
            page_list, remapped_pages = flatten_selected_pages(page_selection, page_count)
            select_pdf_pages(reader, outstream, page_list=page_list)

        doc.binary_representation = outstream.getvalue()
        doc.properties["remapped_pages"] = remapped_pages
        new_elements = filter_elements_by_page(doc.elements, page_list)
        doc.elements = new_elements
        return doc

    return select_pages_fn


def split_pdf(num_pages: int = 1) -> Callable[[Document], list[Document]]:
    """
    Splits a PDF document into smaller documents, each containing a specified number of pages.

    Args:
        num_pages: The number of pages in each split document.

    Returns:
        A function that takes a Document and returns a list of Documents with the specified
        number of pages. Suitable for passing to FlatMap on a DocSet.
    """

    def split_pdf_fn(doc: Document) -> list[Document]:
        """
        Splits a PDF into multiple documents, each containing the specified number of pages.

        This method is suitable for passing to FlatMap on a DocSet.
        """

        if doc.binary_representation is None:
            logging.warning(f"No binary representation found in doc {doc.doc_id}. Skipping splitting.")
            return [doc]

        with PdfReader(BytesIO(doc.binary_representation)) as reader:
            page_count = len(reader.pages)

            new_docs = []

            for idx, start in enumerate(range(1, page_count, num_pages)):
                outstream = BytesIO()
                page_list, remapped_pages = flatten_selected_pages(
                    [[start, min(page_count, start + num_pages - 1)]], page_count
                )
                select_pdf_pages(reader, outstream, page_list=page_list)

                new_elements = filter_elements_by_page(doc.elements, page_list)
                new_doc = Document(binary_representation=outstream.getvalue(), elements=new_elements)
                new_doc.properties["_original_id"] = doc.doc_id
                new_doc.properties["_split_index"] = idx
                new_doc.properties["remapped_pages"] = remapped_pages

                new_docs.append(new_doc)

            return new_docs

    return split_pdf_fn


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


def promote_title(elements: list[Element], title_candidate_elements=["Section-header", "Caption"]) -> list[Element]:
    section_header_big_font = 0
    section_header = None
    for ele in elements:
        if ele.properties["page_number"] != 1:
            continue
        if ele.type == "Title":
            return elements
        else:
            font_size = ele.properties.get("font_size", None)
            if ele.type in title_candidate_elements and font_size and font_size > section_header_big_font:
                section_header_big_font = font_size
                section_header = ele
    if section_header:
        section_header.type = "Title"
    return elements


def get_element_image(element: Element, document: Document) -> Image.Image:
    if isinstance(element, ImageElement) and (im := element.as_image()) is not None:
        return im
    assert document.type == "pdf", "Cannot get picture of element from non-pdf"
    assert document.binary_representation is not None, "Cannot get image since there is not binary representation"
    assert element.bbox is not None, "Cannot get picture of element if it has no BBox"
    assert element.properties.get("page_number") is not None and isinstance(
        element.properties["page_number"], int
    ), "Cannot get picture of element without known page number"
    bits = BytesIO(document.binary_representation)
    pagebits = BytesIO()
    select_pdf_pages(bits, pagebits, [element.properties["page_number"]])
    images = pdf2image.convert_from_bytes(pagebits.getvalue())
    im = crop_to_bbox(images[0], element.bbox)
    return im
