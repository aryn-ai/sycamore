from io import BytesIO
from PIL import Image

from sycamore.data import Document
from sycamore import DocSet, Execution
from sycamore.functions.document import DrawBoxes, split_and_convert_to_image
from sycamore.utils.image_utils import show_images


def show_pages(docset: DocSet, limit: int = 2):
    new_docset = (
        docset.flat_map(split_and_convert_to_image)
        .limit(limit)
        .map_batch(DrawBoxes(), f_constructor_kwargs={"draw_table_cells": True})
    )

    execution = Execution(new_docset.context, new_docset.plan)
    dataset = execution.execute(new_docset.plan)
    documents = [Document.from_row(row) for row in dataset.take(limit)]
    images = [
        Image.open(BytesIO(doc.binary_representation)) for doc in documents if doc.binary_representation is not None
    ]

    show_images(images)
