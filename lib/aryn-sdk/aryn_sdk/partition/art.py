from os import PathLike
from typing import BinaryIO, Union
import io
from packaging.version import InvalidVersion, Version
from pathlib import Path
import PIL
from PIL import Image, ImageDraw, ImageFont
import pdf2image
import logging
import sys


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler(sys.stderr))

COLORS = {
    "table": "tomato",
    "": "orange",  # table cells don't get a label bc it covers up the table label
    "Text": "blue",
    "Image": "green",
    "Section-header": "mediumblue",
    "Page-footer": "dimgrey",
    "List-item": "dodgerblue",
}


# Drawing algorithm, heavily adapted from sycamore.utils.pdf_utils.show_pages
# 1. Convert the pdf to a list of images
# 2. For each element
#     - find the page number it's on
#     - scale the bbox proportionally to the size of the page
#     - add a rectangle to the image of the page
#     - add a text box with the label to the image of the page
#     - if draw_table_cells and this element is a table
#         - do all the bbox/rectangle/textbox steps above for each cell.
def draw_with_boxes(
    pdf_file: Union[PathLike, BinaryIO, str], partitioning_data: dict, draw_table_cells: bool = False
) -> list[Image.Image]:
    """
    Create a list of images from the provided pdf, one for each page, with bounding boxes detected by
    the partitioner drawn on.

    Args:
        pdf_file: an open file or path to a pdf file upon which to draw
        partitioning_data: the output from ``aryn_sdk.partition.partition_file``
        draw_table_cells: whether to draw individually detected cells of tables.
            default: False

    Returns:
        a list of images of pages of the pdf, each with bounding boxes drawn on

    Example:

         .. code-block:: python

            from aryn_sdk.partition import partition_file, draw_with_boxes

            with open("my-favorite-pdf.pdf", "rb") as f:
                data = partition_file(
                    f,
                    aryn_api_key="MY-ARYN-TOKEN",
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True
                )
            pages = draw_with_boxes("my-favorite-pdf.pdf", data, draw_table_cells=True)
    """
    assert (
        "elements" in partitioning_data
    ), 'There are no "elements" in the provided partitioning data, so drawing them is impossible'
    if isinstance(pdf_file, str):
        pdf_file = Path(pdf_file)
    if isinstance(pdf_file, PathLike):
        with open(pdf_file, "rb") as f:
            pdf_file = io.BytesIO(f.read())
    images = pdf2image.convert_from_bytes(pdf_file.read())
    for element in partitioning_data["elements"]:
        if "page_number" in element["properties"]:
            im = images[element["properties"]["page_number"] - 1]
            _draw_box_on_image(im, element)
            if draw_table_cells and element.get("type") == "table" and (table_object := element.get("table")):
                for cell in table_object.get("cells", []):
                    if cell and (bbox := cell.get("bbox")):
                        _draw_box_on_image(
                            im,
                            element={
                                "type": "",
                                "bbox": (
                                    bbox["x1"],
                                    bbox["y1"],
                                    bbox["x2"],
                                    bbox["y2"],
                                ),
                            },
                        )
    return images


def _supports_font_size() -> bool:
    try:
        return Version(PIL.__version__) >= Version("10.1.0")
    except InvalidVersion:
        return False


def _draw_box_on_image(image: Image.Image, element: dict):
    e_coords = element.get("bbox")
    if e_coords is None:
        return
    coords = (
        e_coords[0] * image.width,
        e_coords[1] * image.height,
        e_coords[2] * image.width,
        e_coords[3] * image.height,
    )
    label = element.get("type", "unknown")
    color = _color_for_label(label)
    canvas = ImageDraw.ImageDraw(image)
    canvas.rectangle(coords, outline=color, width=3)
    label_loc = (coords[0] - image.width / 100, coords[1] - image.height / 100)

    if _supports_font_size():
        font = ImageFont.load_default(size=20)
    else:
        font = ImageFont.load_default()

    font_box = canvas.textbbox(label_loc, label, font=font)
    canvas.rectangle(font_box, fill="yellow")
    canvas.text(label_loc, label, fill="black", font=font, align="left")


def _color_for_label(label) -> str:
    if label in COLORS:
        return COLORS[label]
    else:
        return "black"
