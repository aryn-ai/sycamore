from io import BytesIO
from typing import Optional
from sycamore.data.element import TableElement

import pdf2image

from sycamore.data import Document, Element
from sycamore.utils.image_utils import try_draw_boxes
from PIL import Image as PImage, ImageDraw


def split_and_convert_to_image(doc: Document) -> list[Document]:
    """Split a document into individual pages as images and convert them into Document objects.

    This function takes a Document object, which may represent a multi-page document, and splits it into individual
    pages. Each page is converted into an image, and a new Document object is created for each page. The resulting
    list contains these new Document objects, each representing one page of the original document and elements making
    up the page.

    The input Document object should have a binary_representation attribute containing the binary data of the pdf
    document. Each page's elements are preserved in the new Document objects, and page-specific properties
    are updated to reflect the image's size, mode, and page number.

    Args:
        doc: The input Document to split and convert.

    Returns:
        A list of Document objects, each representing a single page of the original document as an image and
        elements making up the page.

    Example:
         .. code-block:: python

            input_doc = Document(binary_representation=pdf_bytes, elements=elements, properties={"author": "John Doe"})
            page_docs = split_and_convert_to_image(input_doc)

    """

    if doc.binary_representation is not None:
        images = pdf2image.convert_from_bytes(doc.binary_representation)
    else:
        return [doc]

    elements_by_page: dict[int, list[Element]] = {}

    for e in doc.elements:
        page_number = e.properties["page_number"]
        elements_by_page.setdefault(page_number, []).append(e)

    sorted_elements_by_page = sorted(elements_by_page.items(), key=lambda x: x[0])
    new_docs = []
    for image, (page, elements) in zip(images, sorted_elements_by_page):
        new_doc = Document(binary_representation=image.tobytes(), elements=elements)
        new_doc.properties.update(doc.properties)
        new_doc.properties.update({"size": list(image.size), "mode": image.mode, "page_number": page})
        new_docs.append(new_doc)
    return new_docs


class DrawBoxes:
    """
    DrawBoxes is a class for adding/drawing boxes around elements within images represented as Document objects.

    This class is designed to enhance Document objects representing images with elements (e.g., text boxes, tables)
    by drawing bounding boxes around each element. It also allows you to customize the color mapping for different
    element types.

    Args:
        font_path: The path to the TrueType font file to be used for labeling.
        default_color: The default color for bounding boxes when the element type is unknown.

    Example:

          .. code-block:: python

            context = sycamore.init()

            font_path="path/to/font.ttf"

            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .flat_map(split_and_convert_to_image)
                .map_batch(DrawBoxes, f_constructor_args=[font_path])
    """

    def __init__(self, font_path: Optional[str] = None, default_color: str = "blue", draw_table_cells: bool = True):
        self.font_path = font_path
        self.color_map = {
            "Title": "red",
            "NarrativeText": "blue",
            "UncategorizedText": "blue",
            "ListItem": "green",
            "table": "orange",
        }
        self.default_color = default_color
        self.draw_table_cells = draw_table_cells

    def _get_color(self, element: Element):
        if element.type is None:
            return self.default_color
        return self.color_map.get(element.type, self.default_color)

    def _draw_boxes(self, doc: Document) -> Document:
        size = tuple(doc.properties["size"])
        mode = doc.properties["mode"]
        image = PImage.frombytes(mode=mode, size=size, data=doc.binary_representation)
        canvas = ImageDraw.Draw(image)

        try_draw_boxes(
            canvas, doc.elements, text_fn=lambda e, _: e.type, color_fn=self._get_color, font_path=self.font_path
        )

        if self.draw_table_cells:
            for e in doc.elements:
                if isinstance(e, TableElement) and e.table is not None:
                    e.table.draw(canvas)

        png_image = BytesIO()
        image.save(png_image, format="PNG")
        doc.binary_representation = png_image.getvalue()
        return doc

    def __call__(self, docs: list[Document]) -> list[Document]:
        return [self._draw_boxes(d) for d in docs]
