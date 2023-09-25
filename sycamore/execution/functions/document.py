from io import BytesIO
from typing import Optional

import pdf2image

from sycamore.data import Document, Element
from PIL import Image as PImage, ImageDraw, ImageFont


def split_and_convert_to_image(doc: Document) -> list[Document]:
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
        new_doc = Document(binary_representation=image.tobytes(), elements={"array": elements})
        new_doc.properties.update(doc.properties)
        new_doc.properties.update({"size": list(image.size), "mode": image.mode, "page_number": page})
        new_docs.append(new_doc)
    return new_docs


class DrawBoxes:
    def __init__(self, font_path: str, default_color: str = "blue"):
        self.font = ImageFont.truetype(font_path, 20)
        self.color_map = {
            "Title": "red",
            "NarrativeText": "blue",
            "UncategorizedText": "blue",
            "ListItem": "green",
            "Table": "orange",
        }
        self.default_color = default_color

    def _get_color(self, e_type: Optional[str]):
        if e_type is None:
            return self.default_color
        return self.color_map.get(e_type, self.default_color)

    def _draw_boxes(self, doc: Document) -> Document:
        size = tuple(doc.properties["size"])
        image_width, image_height = size
        mode = doc.properties["mode"]
        image = PImage.frombytes(mode=mode, size=size, data=doc.binary_representation)
        canvas = ImageDraw.Draw(image)

        for i, e in enumerate(doc.elements):
            layout_width = e.properties["coordinates"]["layout_width"]
            layout_height = e.properties["coordinates"]["layout_height"]

            box = [
                e.properties["coordinates"]["points"][0][0] / layout_width,
                e.properties["coordinates"]["points"][0][1] / layout_height,
                e.properties["coordinates"]["points"][2][0] / layout_width,
                e.properties["coordinates"]["points"][2][1] / layout_height,
            ]

            bbox = [box[0] * image_width, box[1] * image_height, box[2] * image_width, box[3] * image_height]

            canvas.rectangle(bbox, fill=None, outline=self._get_color(e.type), width=3)
            font_box = canvas.textbbox(
                (bbox[0] - image_width / 120, bbox[1] - image_height / 120), str(i + 1), font=self.font
            )
            canvas.rectangle(font_box, fill="yellow")
            canvas.text(
                (bbox[0] - image_width / 120, bbox[1] - image_height / 120),
                str(i + 1),
                fill="black",
                font=self.font,
                align="left",
            )

        png_image = BytesIO()
        image.save(png_image, format="PNG")
        doc.binary_representation = png_image.getvalue()
        return doc

    def __call__(self, docs: list[Document]) -> list[Document]:
        return [self._draw_boxes(d) for d in docs]
