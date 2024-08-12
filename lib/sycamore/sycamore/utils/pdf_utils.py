from io import BytesIO
from PIL import Image

from sycamore import DocSet
from sycamore.functions.document import DrawBoxes, split_and_convert_to_image
from sycamore.utils.image_utils import show_images
from sycamore.data import Document
import json 
from IPython.display import display, HTML

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


def enumerate_images_and_tables(m_pages: list[Document]):
    num_pages = len(m_pages)
    for i in range(0, num_pages):
        m_page = m_pages[i]
        print("Path: ", m_page.properties['path'])
        for e in m_page.elements:
            if e.type == "Image":
                print("Image summary: ", e.properties['summary'], "\n")
                print()
            if e.type == "table":
                table_text_html = e.get("table")
                if table_text_html:
                    display(HTML(table_text_html.to_html()))
                print()

def display_page_and_table_properties(some_pages: list[Document]):
    for m_page in some_pages:
        print("Page props: ")
        display(m_page.properties['entity'])
        print()
        for e in m_page.elements:
            if e and  e.type=="table":
                print("Element Type: ", e.type)
                print("Element Properties: ", json.dumps(e.properties, indent=2, default=str))
                display(HTML(e.text_representation))
